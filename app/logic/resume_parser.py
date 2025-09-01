# ✅ File: app/logic/resume_parser.py

import fitz  # PyMuPDF
import docx2txt
import io
import re
import tempfile
import os
from datetime import datetime
from dateutil import parser as dateparser
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image  # <-- added: needed for robust OCR with pytesseract

# ----------------------------
# SpaCy: load safely with fallback
# ----------------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        _HAS_NER = True
    except Exception:
        # Fall back to a blank English pipeline (tokenization only)
        nlp = spacy.blank("en")
        _HAS_NER = False
except Exception:
    # If spacy is not available at all, create a tiny stub
    class _StubTok:
        def __call__(self, text: str):
            class _Doc:
                ents = []
                def __iter__(self): return iter([])
            return _Doc()
    nlp = _StubTok()  # type: ignore
    _HAS_NER = False

# ----------------------------
# OCR toggle (via env)
# ----------------------------
_ENABLE_OCR = str(os.getenv("ENABLE_OCR", "0")).strip() not in {"", "0", "false", "False"}
try:
    import pytesseract  # optional
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

# ----------------------------
# Load known skills vocabulary
# ----------------------------
SKILL_FILE = Path(__file__).parent.parent / "resources" / "skills.txt"
KNOWN_SKILLS: set = set()

# ✅ NEW: Multi-path fallback loader to avoid silent empty vocab when the file
#        lives outside app/resources (e.g., project root or /resources).
def _load_skills_vocab() -> None:
    global KNOWN_SKILLS
    candidate_paths = [
        SKILL_FILE,
        Path("app/resources/skills.txt"),
        Path("resources/skills.txt"),
        Path("skills.txt"),
    ]
    for p in candidate_paths:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    KNOWN_SKILLS = set(line.strip().lower() for line in f if line.strip())
                if KNOWN_SKILLS:
                    return
        except Exception:
            # try next path silently (non-fatal)
            continue
    # if all fail, keep empty set (degrade gracefully)

_load_skills_vocab()

# ----------------------------
# Helpers
# ----------------------------
def _safe_join(lines: List[str]) -> str:
    return "\n".join([l for l in lines if isinstance(l, str)])

def _strip_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _clean_text_for_blob(text: str) -> str:
    """
    Light normalization for search_blob (non-destructive).
    Keeps alphanumerics and spaces, lowercases, collapses whitespace.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", " ", t)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s+/.,_-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ----------------------------
# Degree & school detectors (additive; for convenience/analytics)
# ----------------------------
_DEGREE_TOKENS = [
    # legal & business
    "llb", "ll.m", "llm", "jd", "bar", "bar-at-law", "bar at law",
    "mba", "bba",
    # generic
    "bs", "b.s", "b.sc", "bsc", "ba", "b.a", "b.tech", "btech",
    "ms", "m.s", "m.sc", "msc", "ma", "m.a", "m.tech", "mtech",
    "phd", "dphil", "doctorate", "bachelors", "masters"
]

def _normalize_degree_string(s: str) -> str:
    return s.replace(".", "").replace("-", "").replace(" ", "").lower()

def _detect_degrees(raw: str) -> List[str]:
    low = raw.lower()
    found = set()
    norm_full = _normalize_degree_string(low)
    for token in _DEGREE_TOKENS:
        t_low = token.lower()
        if t_low in low or _normalize_degree_string(t_low) in norm_full:
            found.add(t_low)
    return sorted(found)

_SCHOOL_PATTERNS = [
    r"\bgraduated from\s+([a-z][a-z0-9&\-\. ]{2,80})",
    r"\bgraduate of\s+([a-z][a-z0-9&\-\. ]{2,80})",
    r"\bfrom\s+([a-z][a-z0-9&\-\. ]{2,80})(?: university| college| law school)\b",
    r"\b([a-z][a-z0-9&\-\. ]{2,80})\s+(?:university|college|law school)\b",
]

def _detect_schools(raw: str) -> List[str]:
    low = raw.lower()
    schools: List[str] = []
    for pat in _SCHOOL_PATTERNS:
        for m in re.finditer(pat, low, flags=re.IGNORECASE):
            sch = m.group(1).strip().lower()
            sch = re.split(r"[.;,]| and | but | however ", sch, flags=re.IGNORECASE)[0]
            sch = re.sub(r"\s+", " ", sch).strip()
            if sch:
                schools.append(sch)
    # de-dup preserve order
    out, seen = [], set()
    for s in schools:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# ----------------------------
# File text extraction helpers
# ----------------------------
def _pdf_text_pymupdf(content: bytes) -> str:
    """
    Primary PDF extractor (fast). Never raises outwards.
    """
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return ""
    try:
        # try empty password auth for some locked PDFs
        try:
            if doc.needs_pass:
                doc.authenticate("")
        except Exception:
            pass
        texts = []
        for page in doc:
            # 'text' is good default; 'blocks' sometimes captures more
            page_text = page.get_text("text") or page.get_text()
            if page_text:
                texts.append(page_text)
        return _safe_join(texts)
    except Exception:
        return ""
    finally:
        try:
            doc.close()
        except Exception:
            pass

def _pdf_text_ocr_with_fitz(content: bytes) -> str:
    """
    OCR fallback using PyMuPDF rasterization + pytesseract.
    Requires ENABLE_OCR=1 and pytesseract installed.
    Never raises outwards; returns "" if anything fails.
    """
    if not (_ENABLE_OCR and _HAS_TESS):
        return ""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return ""
    try:
        texts = []
        zoom = 2.0  # ~144 DPI
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                # ✅ ensure pytesseract receives a Pillow Image (not a BytesIO)
                img = Image.open(io.BytesIO(img_bytes)).convert("L")
                txt = pytesseract.image_to_string(img)
                if txt:
                    texts.append(txt)
            except Exception:
                continue
        return _safe_join(texts)
    except Exception:
        return ""
    finally:
        try:
            doc.close()
        except Exception:
            pass

def extract_text_from_pdf(content: bytes) -> str:
    """
    Try text extraction; if blank and OCR enabled, OCR fallback.
    """
    text = _pdf_text_pymupdf(content)
    if _strip_text(text):
        return text
    # OCR fallback
    ocr_text = _pdf_text_ocr_with_fitz(content)
    return ocr_text

def extract_text_from_docx(content: bytes) -> str:
    """
    Use a temp file path for docx2txt to avoid file-like compatibility issues.
    Never raises outwards.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
            tmp.write(content)
            tmp.flush()
            t = docx2txt.process(tmp.name) or ""
            return t
    except Exception:
        # minimal salvage: try naive decode
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def extract_text_from_doc_legacy(content: bytes) -> str:
    """
    Simple fallback for legacy .doc files.
    Never raises outwards.
    """
    try:
        raw = content.decode("latin-1", errors="ignore")
    except Exception:
        raw = str(content)

    # Remove control chars
    raw = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", raw)

    # Keep readable sequences
    pieces = re.findall(r"[A-Za-z0-9@#\+\-_/.,:;()&% ]{3,}", raw)
    text = " ".join(pieces)

    # Normalize whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ----------------------------
# Core field extractors
# ----------------------------
def extract_name(text: str) -> Optional[str]:
    # 1) NER-based (PERSON)
    if _HAS_NER:
        try:
            doc = nlp(text)
            names = [
                ent.text.strip()
                for ent in doc.ents
                if getattr(ent, "label_", "") == "PERSON" and 2 <= len(ent.text.strip().split()) <= 4
            ]
            if names:
                return names[0]
        except Exception:
            pass

    # 2) Top lines regex
    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+$", line):
            return line

    # 3) “Name: …” pattern
    match = re.search(r"(?i)^Name[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)+)", text)
    return match.group(1) if match else None

def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group() if match else None

def extract_phone(text: str) -> Optional[str]:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group() if match else None

def extract_skills(text: str) -> List[str]:
    """
    Skills extraction:
      1) Phrase-level contains check from skills.txt (multi-word first)
      2) Token-level fallback when vocab present
    """
    low = text.lower()
    matched = set()

    # Phrase contains (longest first)
    if KNOWN_SKILLS:
        for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
            try:
                if skill and skill in low:
                    matched.add(skill)
            except Exception:
                continue

    # Token-level fallback
    try:
        doc = nlp(low)
        tokens = set(
            getattr(t, "text", "") for t in doc
            if getattr(t, "is_stop", False) is False and getattr(t, "is_punct", False) is False
        )
        matched |= (KNOWN_SKILLS.intersection(tokens))
    except Exception:
        pass

    return sorted({m.strip().lower() for m in matched if m and m.strip()})

def _round_experience_value(years: float) -> float:
    """
    Round to 1 decimal; clamp negatives; reduce tiny noise.
    """
    try:
        if years < 0:
            years = 0.0
        # snap very small positive noise to 0
        if 0 < years < 0.05:
            years = 0.0
        return round(float(years), 1)
    except Exception:
        return 0.0

def _format_experience_display(years: float) -> str:
    """
    Pretty display string without '0.' or trailing '.0' issues.
    """
    y = _round_experience_value(years)
    if abs(y - int(y)) < 1e-9:
        y_int = int(y)
        return f"{y_int} year" if y_int == 1 else f"{y_int} years"
    return f"{y} years"

def extract_experience(text: str) -> float:
    """
    Estimate total years of experience using two strategies:
      A) Parse explicit "X years experience" mentions (max value wins)
      B) Sum date ranges like "Jan 2020 - Mar 2023" or "May 2019 to Present"

    Tweaks:
      - tolerate forms like '0. years experience' (missing decimal digit)
      - clamp negatives; return rounded 1-decimal
    """
    # A) Explicit mentions (anchor to 'experience' to avoid random numbers)
    try:
        # allow optional decimals with missing fractional digits (e.g., '0. years')
        year_match = re.findall(
            r"(\d+(?:\.\d*)?)\s*\+?\s*(?:years|yrs)[\s\w]*experience",
            text,
            re.IGNORECASE
        )
        explicit_years = 0.0
        for m in year_match or []:
            try:
                # '0.' -> float('0.') works; but guard just in case
                val = float(m if m[-1].isdigit() else m.rstrip(".") or "0")
                explicit_years = max(explicit_years, val)
            except Exception:
                continue
    except Exception:
        explicit_years = 0.0

    # B) Date ranges
    try:
        date_ranges = re.findall(
            r'(\w+\s\d{4})\s*(?:-|–|—|to)\s*(\w+\s\d{4}|present)',
            text,
            re.IGNORECASE
        )
        total_months = 0
        for start, end in date_ranges:
            try:
                start_date = dateparser.parse(start)
                end_date = datetime.now() if re.search(r"present", end, re.IGNORECASE) else dateparser.parse(end)
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_months += max(months, 0)
            except Exception:
                continue
        calculated_years = total_months / 12.0
    except Exception:
        calculated_years = 0.0

    years = max(explicit_years, calculated_years)
    return _round_experience_value(years)

def extract_projects(text: str) -> List[Dict[str, Any]]:
    """
    Extract up to 5 projects from recognizable sections, tagging technologies
    using the skills vocabulary.
    """
    try:
        raw_sections = re.findall(
            r'(?:Project[s]?:?|Responsibilities:|Description:)[\s\S]{0,800}',
            text,
            re.IGNORECASE
        )
    except Exception:
        raw_sections = []

    projects: List[Dict[str, Any]] = []
    seen = set()

    for section in raw_sections:
        lines = [line.strip() for line in section.strip().split('\n') if len(line.strip()) > 20]
        if not lines:
            continue

        name = lines[0][:80]
        description = " ".join(lines[1:]).strip()
        if not name or name in seen:
            continue

        seen.add(name)

        tech: List[str] = []
        low_desc = description.lower()
        if KNOWN_SKILLS:
            for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
                if skill and skill in low_desc:
                    tech.append(skill)

        projects.append({
            "name": name,
            "description": description,
            "tech": sorted(list({t for t in tech}))
        })

        if len(projects) >= 5:
            break

    return projects

def extract_summary(text: str) -> str:
    try:
        match = re.search(
            r"(Summary|Objective)[:\n\s]*(.*?)(?:\n\n|\n[A-Z][a-z]+:|\n[A-Z ]{3,}|Education|Experience|Projects|Skills|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(2).strip()[:600]
    except Exception:
        pass

    # NLP fallback: first paragraph with >=2 verbs (when NER pipeline available)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) > 10]
    for para in paragraphs:
        try:
            doc = nlp(para)
            verbs = [t for t in getattr(doc, "__iter__", lambda: [])() if getattr(t, "pos_", "") == "VERB"]
            if len(verbs) >= 2:
                return para.strip()[:600]
        except Exception:
            break
    return ""

def extract_education(text: str) -> list:
    try:
        sections = re.findall(r"(Bachelor|Master|B\.Tech|M\.Tech|BSc|MSc|Ph\.D)[\s\S]{0,200}", text, re.IGNORECASE)
    except Exception:
        sections = []
    results = []
    for section in sections:
        match = re.search(r"(?P<degree>[A-Za-z .]+)[,|\n\s]*(?P<school>[A-Za-z ]+)[,|\n\s]*(?P<year>\d{4})", section)
        if match:
            results.append({
                "degree": match.group("degree").strip(),
                "school": match.group("school").strip(),
                "year": match.group("year"),
                "gpa": "N/A"
            })
    return results

def extract_work_history(text: str) -> list:
    try:
        jobs = re.findall(r"(?:Experience|Work)[\s\S]{0,1000}", text, re.IGNORECASE)
    except Exception:
        jobs = []
    results = []
    for job in jobs:
        lines = [line.strip() for line in job.strip().split('\n') if line.strip()]
        for i in range(0, len(lines) - 2, 3):
            title = lines[i]
            company = lines[i + 1]
            duration = lines[i + 2]
            if len(title) > 2 and len(company) > 2 and len(duration) > 2:
                results.append({
                    "title": title,
                    "company": company,
                    "duration": duration,
                    "description": "Worked on various tasks and contributed to team goals."
                })
    return results

def extract_location(text: str) -> str:
    if _HAS_NER:
        try:
            doc = nlp(text)
            locations = [ent.text.strip() for ent in doc.ents if getattr(ent, "label_", "") == "GPE"]
            if locations:
                return locations[0]
        except Exception:
            pass

    match = re.search(r"\b(?:located in|from|based in)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text)
    return match.group(1) if match else "N/A"

# ----------------------------
# Master extractor
# ----------------------------
def extract_info(text: str) -> dict:
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)
    experience = extract_experience(text)          # float years (rounded 1-decimal)
    projects = extract_projects(text)
    summary = extract_summary(text)
    education = extract_education(text)
    workHistory = extract_work_history(text)
    location = extract_location(text)

    # derive current role/title from work history or summary
    current_title = workHistory[0]["title"] if workHistory else None
    company = workHistory[0]["company"] if workHistory else None

    # Small project blob for downstream filters
    project_summary_text = " ".join(
        [f"{p.get('name','')} {p.get('description','')}" for p in projects]
    ).strip()

    experience_float = float(experience)
    experience_display = _format_experience_display(experience_float)  # ✅ clean display
    experience_rounded = _round_experience_value(experience_float)     # ✅ numeric rounded

    # Additive detections (for strong filtering convenience)
    degrees_detected = _detect_degrees(text or "")
    schools_detected = _detect_schools(text or "")

    info = {
        "name": name or "No Name",                 # ✅ fallback
        "email": email or None,
        "phone": phone or None,

        # Skills
        "skills": skills or [],
        "skills_text": (summary or "")[:300],      # small blob to help skill pickup if needed

        # Experience (multiple aliases so any consumer can read it)
        "experience": experience_float,
        "total_experience_years": experience_float,
        "years_of_experience": experience_float,
        "experience_years": experience_float,
        "yoe": experience_float,
        # Additive nice-to-have fields to fix UI like "0. years"
        "experience_rounded": experience_rounded,  # float (1-decimal)
        "experience_display": experience_display,  # string ("2 years", "0.5 years", "1 year")

        # Title/Role (multiple aliases)
        "title": (current_title or "N/A"),
        "job_title": (current_title or "N/A"),
        "current_title": (current_title or "N/A"),

        # Projects
        "projects": projects,
        "project_summary": project_summary_text,
        "project_details": project_summary_text,
        "portfolio": project_summary_text,

        # Location
        "location": location,

        # Raw text
        "raw_text": text,

        # Nested resume section
        "resume": {
            "summary": summary,
            "education": education,
            "workHistory": workHistory,
            "projects": projects
        },

        # Convenience fields
        "currentRole": current_title or "N/A",
        "company": company or "N/A",

        # Additive: strong-filter helpers (education)
        "degrees_detected": degrees_detected,
        "schools_detected": schools_detected,
    }

    # ----------------------------
    # NEW: Precompute search_blob for fast ANN/text recall
    # ----------------------------
    # Compose a compact, normalized blob using salient fields.
    # This is non-breaking (just an extra field).
    try:
        parts: List[str] = []
        parts.append(info.get("name") or "")
        parts.append(info.get("currentRole") or info.get("title") or "")
        parts.append(info.get("location") or "")
        # skills
        if info.get("skills"):
            parts.append(" ".join([s for s in info["skills"] if isinstance(s, str)]))
        # projects
        if projects:
            proj_bits: List[str] = []
            for p in projects[:6]:
                if isinstance(p, dict):
                    proj_bits.append(str(p.get("name") or ""))
                    proj_bits.append(str(p.get("description") or ""))
                elif isinstance(p, str):
                    proj_bits.append(p)
            parts.append(" ".join([x for x in proj_bits if x]))
        # summary (short)
        if summary:
            parts.append(summary[:600])
        # total experience (numeric)
        parts.append(str(experience_float))
        # raw text (truncate to keep blob compact)
        raw = text or ""
        truncate_chars = int(os.getenv("SEARCH_BLOB_TRUNCATE_CHARS", "4000"))
        if raw:
            parts.append(raw[:truncate_chars])

        search_blob = _clean_text_for_blob(" ".join([p for p in parts if p]).strip())
        info["search_blob"] = search_blob
    except Exception:
        # Safe fallback: at least include raw_text cleaned
        try:
            info["search_blob"] = _clean_text_for_blob((text or "")[:4000])
        except Exception:
            info["search_blob"] = ""

    return info

def parse_resume_file(filename: str, content: bytes) -> dict:
    """
    Never lets exceptions bubble up. If everything fails, returns extract_info("")
    so caller can treat as 'empty' rather than a hard 'parse_error'.
    """
    try:
        lower = (filename or "").lower()
        text = ""
        if lower.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif lower.endswith(".docx"):
            text = extract_text_from_docx(content)
        elif lower.endswith(".doc"):
            text = extract_text_from_doc_legacy(content)
        else:
            # Unknown; try docx first, then legacy
            try:
                text = extract_text_from_docx(content)
            except Exception:
                text = extract_text_from_doc_legacy(content)
        return extract_info(text or "")
    except Exception:
        # last-resort safe return
        return extract_info("")
