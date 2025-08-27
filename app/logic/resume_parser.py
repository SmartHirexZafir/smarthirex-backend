# ✅ File: app/logic/resume_parser.py

import fitz  # PyMuPDF
import docx2txt
import io
import re
import tempfile
from datetime import datetime
from dateutil import parser as dateparser
from pathlib import Path
from typing import List, Dict, Any, Optional

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
# Load known skills vocabulary
# ----------------------------
SKILL_FILE = Path(__file__).parent.parent / "resources" / "skills.txt"
KNOWN_SKILLS: set = set()
if SKILL_FILE.exists():
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        KNOWN_SKILLS = set(line.strip().lower() for line in f if line.strip())


# ----------------------------
# File text extraction helpers
# ----------------------------
def extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts)
    finally:
        doc.close()


def extract_text_from_docx(content: bytes) -> str:
    """
    Use a temp file path for docx2txt to avoid file-like compatibility issues.
    """
    with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
        tmp.write(content)
        tmp.flush()
        return docx2txt.process(tmp.name) or ""


def extract_text_from_doc_legacy(content: bytes) -> str:
    """
    Simple fallback for legacy .doc files.
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
        doc = nlp(text)
        names = [
            ent.text.strip()
            for ent in doc.ents
            if getattr(ent, "label_", "") == "PERSON" and 2 <= len(ent.text.strip().split()) <= 4
        ]
        if names:
            return names[0]

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
            if skill and skill in low:
                matched.add(skill)

    # Token-level fallback
    try:
        doc = nlp(low)
        tokens = set(
            getattr(t, "text", "") for t in doc
            if getattr(t, "is_stop", False) is False and getattr(t, "is_punct", False) is False
        )
        matched |= (KNOWN_SKILLS.intersection(tokens))
    except Exception:
        # If spaCy disabled, skip token fallback
        pass

    return sorted({m.strip().lower() for m in matched if m and m.strip()})


def extract_experience(text: str) -> float:
    """
    Estimate total years of experience using two strategies:
      A) Parse explicit "X years experience" mentions (max value wins)
      B) Sum date ranges like "Jan 2020 - Mar 2023" or "May 2019 to Present"
    """
    # A) Explicit mentions (anchor to 'experience' to avoid random numbers)
    year_match = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)[\s\w]*experience",
        text,
        re.IGNORECASE
    )
    explicit_years = max(map(float, year_match)) if year_match else 0.0

    # B) Date ranges
    # Use non-capturing group for separators: -, – (en-dash), — (em-dash), to
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
    calculated_years = round(total_months / 12, 1)

    return float(max(explicit_years, calculated_years))


def extract_projects(text: str) -> List[Dict[str, Any]]:
    """
    Extract up to 5 projects from recognizable sections, tagging technologies
    using the skills vocabulary.
    """
    raw_sections = re.findall(
        r'(?:Project[s]?:?|Responsibilities:|Description:)[\s\S]{0,800}',
        text,
        re.IGNORECASE
    )
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
    match = re.search(
        r"(Summary|Objective)[:\n\s]*(.*?)(?:\n\n|\n[A-Z][a-z]+:|\n[A-Z ]{3,}|Education|Experience|Projects|Skills|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(2).strip()[:600]

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
    sections = re.findall(r"(Bachelor|Master|B\.Tech|M\.Tech|BSc|MSc|Ph\.D)[\s\S]{0,200}", text, re.IGNORECASE)
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
    jobs = re.findall(r"(?:Experience|Work)[\s\S]{0,1000}", text, re.IGNORECASE)
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
    experience = extract_experience(text)          # float years
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

    return {
        "name": name or "No Name",                 # ✅ fallback
        "email": email or None,
        "phone": phone or None,

        # Skills
        "skills": skills or [],
        "skills_text": (summary or "")[:300],      # small blob to help skill pickup if needed

        # Experience (multiple aliases so any consumer can read it)
        "experience": experience_float,                 # original key
        "total_experience_years": experience_float,     # primary for filters
        "years_of_experience": experience_float,        # alias
        "experience_years": experience_float,           # alias
        "yoe": experience_float,                        # alias

        # Title/Role (multiple aliases)
        "title": (current_title or "N/A"),
        "job_title": (current_title or "N/A"),
        "current_title": (current_title or "N/A"),

        # Projects
        "projects": projects,                           # structured list
        "project_summary": project_summary_text,        # blob for project keyword filter
        "project_details": project_summary_text,        # alias
        "portfolio": project_summary_text,              # alias

        # Location
        "location": location,

        # Raw text
        "raw_text": text,

        # Nested resume section (unchanged)
        "resume": {
            "summary": summary,
            "education": education,
            "workHistory": workHistory,
            "projects": projects
        },

        # Convenience fields
        "currentRole": current_title or "N/A",
        "company": company or "N/A",
    }


def parse_resume_file(filename: str, content: bytes) -> dict:
    lower = (filename or "").lower()
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

    return extract_info(text)
