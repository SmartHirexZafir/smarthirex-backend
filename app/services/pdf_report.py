# app/services/pdf_report.py
"""
Lightweight PDF report builder for a candidate's test attempt.

- Tries to use reportlab if available (nice layout).
- Falls back to a tiny hand-rolled PDF (valid single-page) if reportlab
  is not installed, so you still get a browser-openable PDF without
  adding deps.

Usage (from a router):
    from app.services.pdf_report import build_test_result_pdf

    pdf_bytes = await build_test_result_pdf(candidate_doc, attempt_doc)
"""

from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.utils.datetime_serialization import serialize_utc


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _iso(dt: Any) -> str:
    if isinstance(dt, datetime):
        try:
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return serialize_utc(dt)
    return str(dt or "")


def _attempt_summary(candidate: Dict[str, Any], attempt: Dict[str, Any]) -> Dict[str, Any]:
    name = _coalesce(candidate.get("name"), "Unknown")
    email = _coalesce(candidate.get("resume", {}).get("email"), candidate.get("email"), "N/A")
    role = _coalesce(candidate.get("predicted_role"), candidate.get("category"), "N/A")

    # scores
    test_score = None
    for key in ("test_score", "testScore", "score", "total_score"):
        v = attempt.get(key) if key in attempt else candidate.get(key)
        if isinstance(v, (int, float)):
            test_score = round(float(v))
            break

    match_score = None
    for key in ("score", "match_score"):
        v = candidate.get(key)
        if isinstance(v, (int, float)):
            match_score = round(float(v))
            break

    avg = None
    if isinstance(test_score, (int, float)) and isinstance(match_score, (int, float)):
        avg = round((test_score + match_score) / 2)

    return {
        "candidate_name": name,
        "candidate_email": email,
        "role": role,
        "attempt_id": str(_coalesce(attempt.get("_id"), attempt.get("id"), "")),
        "submitted_at": _iso(_coalesce(attempt.get("submittedAt"), attempt.get("created_at"), attempt.get("ts"))),
        "test_score": test_score,
        "match_score": match_score,
        "avg_score": avg,
    }


def _shape_questions(attempt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize question details from various graders/runners into a flat list.

    Expected sources:
      - evaluate_test(...): details: [{ question, type, awarded, max_points, ... }]
      - code runner: results/tests list with passed/score/etc.
    """
    qs: List[Dict[str, Any]] = []

    details = attempt.get("details") or attempt.get("results") or []
    if isinstance(details, dict) and "details" in details:
        details = details.get("details")  # guard

    if not isinstance(details, list):
        return qs

    for i, item in enumerate(details):
        if not isinstance(item, dict):
            continue
        q_text = item.get("question") or item.get("name") or f"Question {i+1}"
        q_type = item.get("type") or item.get("kind") or "MCQ"
        awarded = _coalesce(item.get("awarded"), item.get("score"), item.get("points"), item.get("awarded_points"))
        maxp = _coalesce(item.get("max_points"), item.get("max"), item.get("out_of"))
        correct = item.get("correct")
        feedback = item.get("feedback") or item.get("explanation")
        qs.append(
            {
                "title": q_text,
                "qtype": q_type,
                "awarded": awarded,
                "max": maxp,
                "correct": correct,
                "feedback": feedback,
            }
        )
    return qs


async def build_test_result_pdf(candidate: Dict[str, Any], attempt: Dict[str, Any]) -> bytes:
    """
    Returns a complete PDF (bytes) representing the attempt.
    """
    # Prefer reportlab if present
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors

        meta = _attempt_summary(candidate, attempt)
        qs = _shape_questions(attempt)

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, title="Assessment Result")
        styles = getSampleStyleSheet()
        body: List[Any] = []

        # Header
        body.append(Paragraph("<b>Assessment Result</b>", styles["Title"]))
        body.append(Spacer(1, 12))
        header_lines = [
            f"Candidate: <b>{meta['candidate_name']}</b>",
            f"Email: {meta['candidate_email']}",
            f"Role: {meta['role']}",
            f"Attempt ID: {meta['attempt_id']}",
            f"Submitted: {meta['submitted_at']}",
        ]
        for line in header_lines:
            body.append(Paragraph(line, styles["Normal"]))
        body.append(Spacer(1, 10))

        # Score block
        parts = []
        if meta["match_score"] is not None:
            parts.append(f"Match Score: <b>{meta['match_score']}%</b>")
        if meta["test_score"] is not None:
            parts.append(f"Test Score: <b>{meta['test_score']}%</b>")
        if meta["avg_score"] is not None:
            parts.append(f"Assessment Score (avg): <b>{meta['avg_score']}%</b>")
        if parts:
            body.append(Paragraph(" • ".join(parts), styles["Heading4"]))
            body.append(Spacer(1, 6))

        # Questions table
        if qs:
            table_data = [["#", "Question", "Type", "Score", "Correct", "Feedback"]]
            for i, q in enumerate(qs, 1):
                sc = ""
                if q["awarded"] is not None and q["max"] is not None:
                    sc = f"{q['awarded']}/{q['max']}"
                elif q["awarded"] is not None:
                    sc = str(q["awarded"])
                table_data.append(
                    [
                        str(i),
                        q["title"],
                        q["qtype"],
                        sc,
                        "✓" if q["correct"] else ("✗" if q["correct"] is False else ""),
                        (q["feedback"] or "")[:250],
                    ]
                )

            tbl = Table(table_data, colWidths=[24, 240, 60, 60, 40, 140])
            tbl.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            body.append(tbl)

        doc.build(body)
        return buf.getvalue()

    except Exception:
        # --- Fallback: minimal single-page PDF with basic text ---
        meta = _attempt_summary(candidate, attempt)
        lines = [
            "Assessment Result",
            f"Candidate: {meta['candidate_name']}",
            f"Email: {meta['candidate_email']}",
            f"Role: {meta['role']}",
            f"Attempt ID: {meta['attempt_id']}",
            f"Submitted: {meta['submitted_at']}",
        ]
        if meta["match_score"] is not None:
            lines.append(f"Match Score: {meta['match_score']}%")
        if meta["test_score"] is not None:
            lines.append(f"Test Score: {meta['test_score']}%")
        if meta["avg_score"] is not None:
            lines.append(f"Assessment Score (avg): {meta['avg_score']}%")

        # very small PDF writer (courier 12pt, simple Tj commands)
        def _pdf_escape(s: str) -> str:
            return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        y = 800
        content = ["BT /F1 12 Tf 50 {} Td ({}) Tj ET".format(y, _pdf_escape(lines[0]))]
        y -= 24
        for ln in lines[1:]:
            content.append("BT /F1 11 Tf 50 {} Td ({}) Tj ET".format(y, _pdf_escape(ln)))
            y -= 18

        stream = "\n".join(content).encode("latin-1", "ignore")
        xref_pos = 0

        def obj(idx: int, body: bytes) -> bytes:
            return f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"

        objects: List[bytes] = []
        objects.append(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))
        objects.append(obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
        objects.append(
            obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
        )
        objects.append(obj(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"))
        objects.append(obj(5, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"))

        # Assemble xref
        output = BytesIO()
        output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        pos = output.tell()
        for o in objects:
            offsets.append(pos)
            output.write(o)
            pos = output.tell()

        xref_pos = pos
        output.write(b"xref\n0 %d\n" % (len(objects) + 1))
        output.write(b"0000000000 65535 f \n")
        for off in offsets:
            output.write(("{:010} 00000 n \n".format(off)).encode())
        output.write(b"trailer\n")
        output.write(
            b"<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref_pos)
        )
        return output.getvalue()
