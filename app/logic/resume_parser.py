# app/logic/resume_parser.py

import fitz  # PyMuPDF
import docx2txt
import io
import re
import tempfile
import os
import json
import zipfile
from xml.etree import ElementTree as ET
from datetime import datetime
from dateutil import parser as dateparser
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image  # needed for robust OCR with pytesseract

# ========= Normalization helpers (shared across ingest/search) =========
from app.logic.normalize import (
    normalize_role,
    normalize_tokens,
    to_years_of_experience,
    norm_text,
    normalize_degree,
    make_search_blob,
    normalize_location,
    STOPWORDS,
)

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

# ✅ Multi-path fallback loader to avoid silent empty vocab when the file
#   lives outside app/resources (e.g., project root or /resources).
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
# Role/Category keyword lexicon (for stable role prediction)
# ----------------------------
# We try to load from JSON so it's data-driven (no hardcoding requirement).
# If absent, we fall back to a safe built-in map (minimal, non-exhaustive).
_ROLE_CAT_JSON_CANDIDATES = [
    Path("app/resources/role_category_keywords.json"),
    Path("resources/role_category_keywords.json"),
    Path("role_category_keywords.json"),
]

ROLE_KEYWORDS: Dict[str, List[str]] = {}
CATEGORY_KEYWORDS: Dict[str, List[str]] = {}

# Built-in light fallback (covers your dataset’s categories & some roles)
_FALLBACK_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Accountant": ["accounting", "accounts payable", "accounts receivable", "ledger", "bookkeeping", "audit", "tax"],
    "Advocate": ["advocate", "attorney", "lawyer", "litigation", "legal research", "barrister", "llb", "llm"],
    "Agriculture": ["agriculture", "agribusiness", "crop", "irrigation", "soil"],
    "Apparel": ["textile", "apparel", "garment", "fashion"],
    "Architecture": ["architect", "architecture", "autocad", "revit"],
    "Arts": ["artist", "illustration", "fine arts", "painting"],
    "Automobile": ["automobile", "automotive", "vehicle", "oem", "autocad"],
    "Aviation": ["aviation", "aircraft", "airline", "aerospace"],
    "Banking": ["bank", "banking", "loan", "credit", "npa", "branch"],
    "Blockchain": ["blockchain", "smart contract", "web3", "solidity", "ethereum"],
    "BPO": ["bpo", "call center", "voice process", "back office", "csr"],
    "Building and Construction": ["construction", "site engineer", "contractor", "civil work"],
    "Business Analyst": ["business analyst", "requirements", "brd", "process modeling", "stakeholder"],
    "Civil Engineer": ["civil engineer", "civil engineering", "autocad", "estimation", "structure"],
    "Consultant": ["consultant", "consulting", "advisory", "strategy"],
    "Data Science": ["data science", "machine learning", "deep learning", "modeling", "python", "pandas"],
    "Database": ["database", "sql server", "postgres", "oracle dba", "mysql"],
    "Designing": ["designer", "ui", "ux", "photoshop", "figma", "adobe"],
    "DevOps": ["devops", "sre", "ci/cd", "kubernetes", "docker", "terraform"],
    "Digital Media": ["digital marketing", "seo", "sem", "social media", "google ads"],
    "DotNet Developer": [".net", "dotnet", "c#", "asp.net"],
    "Education": ["teacher", "lecturer", "professor", "education"],
    "Electrical Engineering": ["electrical", "electronics", "circuit", "power systems"],
    "ETL Developer": ["etl", "data pipeline", "ingestion", "ssis", "informatica"],
    "Finance": ["finance", "financial analysis", "fp&a", "valuation"],
    "Food and Beverages": ["food", "beverage", "f&b", "chef", "kitchen"],
    "Health and Fitness": ["fitness", "trainer", "nutrition", "health"],
    "Human Resources": ["human resources", "hr", "recruitment", "talent acquisition", "payroll"],
    "Information Technology": ["information technology", "it", "software", "infrastructure"],
    "Java Developer": ["java", "spring", "j2ee", "hibernate"],
    "Management": ["manager", "management", "leadership", "pmo"],
    "Mechanical Engineer": ["mechanical", "cad", "manufacturing", "solidworks"],
    "Network Security Engineer": ["network security", "firewall", "siem", "ids", "ips"],
    "Operations Manager": ["operations manager", "operations management", "sop"],
    "PMO": ["pmo", "project management office", "governance"],
    "Public Relations": ["public relations", "pr", "media relations"],
    "Python Developer": ["python", "django", "flask", "fastapi"],
    "React Developer": ["react", "next", "javascript", "redux", "frontend"],
    "Sales": ["sales", "business development", "lead generation", "crm"],
    "SAP Developer": ["sap", "abap", "sap hana", "sap fico", "sap mm"],
    "SQL Developer": ["sql", "stored procedures", "pl/sql", "t-sql"],
    "Testing": ["qa", "testing", "selenium", "automation", "jmeter"],
    "Web Designing": ["web designer", "html", "css", "javascript", "ui/ux"],
}
_FALLBACK_ROLE_KEYWORDS: Dict[str, List[str]] = {
    # More specific roles for better card labeling
    "Data Scientist": ["data science", "machine learning", "modeling", "pytorch", "tensorflow"],
    "ML Engineer": ["ml engineer", "mlops", "deployment", "tensorflow", "pytorch"],
    "Frontend Developer": ["react", "next", "vue", "angular", "javascript", "typescript", "frontend"],
    "Backend Developer": ["node", "django", "flask", "spring", "backend", "api development"],
    "Full Stack Developer": ["full-stack", "mern", "mean", "django react", "node react"],
    "React Developer": ["react", "redux", "next"],
    "Python Developer": ["python", "django", "flask", "fastapi"],
    "Java Developer": ["java", "spring", "hibernate"],
    "DevOps Engineer": ["devops", "sre", "kubernetes", "docker", "terraform", "ci/cd"],
    "ETL Developer": ["etl", "ssis", "informatica", "data pipeline"],
    "QA Engineer": ["qa", "testing", "selenium", "automation"],
    "Business Analyst": ["business analyst", "requirements", "brd", "stakeholder"],
    "HR Manager": ["hr", "human resources", "recruitment", "payroll"],
    "Operations Manager": ["operations manager", "operations"],
    "Network Security Engineer": ["network security", "firewall", "siem"],
    "SAP Developer": ["sap", "abap", "hana"],
    "SQL Developer": ["sql", "t-sql", "pl/sql", "database"],
    "Accountant": ["accounting", "ledger", "bookkeeping", "audit"],
    "Advocate": ["advocate", "attorney", "lawyer", "litigation"],
}

def _load_role_category_keywords() -> None:
    """Load ROLE_KEYWORDS and CATEGORY_KEYWORDS from JSON if present."""
    global ROLE_KEYWORDS, CATEGORY_KEYWORDS
    for p in _ROLE_CAT_JSON_CANDIDATES:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rk = data.get("role_keywords") or data.get("roles") or {}
                ck = data.get("category_keywords") or data.get("categories") or {}
                if isinstance(rk, dict) and rk:
                    ROLE_KEYWORDS = {str(k): [str(x).lower() for x in (v or []) if isinstance(x, (str,))] for k, v in rk.items()}
                if isinstance(ck, dict) and ck:
                    CATEGORY_KEYWORDS = {str(k): [str(x).lower() for x in (v or []) if isinstance(x, (str,))] for k, v in ck.items()}
                break
        except Exception:
            # ignore and try next
            continue
    # fallbacks if not loaded
    if not CATEGORY_KEYWORDS:
        CATEGORY_KEYWORDS = _FALLBACK_CATEGORY_KEYWORDS
    if not ROLE_KEYWORDS:
        ROLE_KEYWORDS = _FALLBACK_ROLE_KEYWORDS

_load_role_category_keywords()

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

def _pdf_text_pypdf(content: bytes) -> str:
    """
    Optional PDF extractor using pypdf / PyPDF2. Never raises outwards.
    """
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))
        texts: List[str] = []
        for page in getattr(reader, "pages", []):
            try:
                t = page.extract_text() or ""
                if t:
                    texts.append(t)
            except Exception:
                continue
        return _safe_join(texts)
    except Exception:
        return ""

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
                # ensure pytesseract receives a Pillow Image (not a BytesIO)
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
    Try pypdf extraction first; if blank, fall back to PyMuPDF; if still blank
    and OCR is enabled, OCR fallback.
    """
    # 1) pypdf / PyPDF2
    text = _pdf_text_pypdf(content)
    if _strip_text(text):
        return text
    # 2) PyMuPDF
    text = _pdf_text_pymupdf(content)
    if _strip_text(text):
        return text
    # 3) OCR fallback
    ocr_text = _pdf_text_ocr_with_fitz(content)
    return ocr_text

def _docx_text_via_zip(content: bytes) -> str:
    """
    Robust DOCX extractor: read word/document.xml from the DOCX (zip) and
    collect all <w:t> text nodes. Never raises outwards.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            with zf.open("word/document.xml") as f:
                xml_bytes = f.read()
        try:
            root = ET.fromstring(xml_bytes)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts = [t.text for t in root.findall(".//w:t", ns) if t is not None and t.text]
            return _strip_text("\n".join(texts))
        except Exception:
            # Regex strip as last resort if XML parser chokes
            raw = re.sub(rb"<[^>]+>", b" ", xml_bytes)
            try:
                return _strip_text(raw.decode("utf-8", errors="ignore"))
            except Exception:
                return _strip_text(raw.decode("latin-1", errors="ignore"))
    except Exception:
        return ""

def extract_text_from_docx(content: bytes) -> str:
    """
    Use a temp file path for docx2txt to avoid file-like compatibility issues.
    If the result is blank or junky, fall back to zip-based XML text extraction.
    Never raises outwards.
    """
    # 1) Try docx2txt
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
            tmp.write(content)
            tmp.flush()
            t = docx2txt.process(tmp.name) or ""
            if _strip_text(t):
                return t
    except Exception:
        pass

    # 2) Fallback: read word/document.xml from the docx zip
    xml_text = _docx_text_via_zip(content)
    if _strip_text(xml_text):
        return xml_text

    # 3) Minimal salvage: try naive decode
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
# Core field extractors (enhanced for Req #8)
# ----------------------------

# Heuristic blocklists to avoid mistaking skills for names
_NAME_BAD_TOKENS = {
    "resume", "curriculum", "vitae", "cv",
    "engineer", "developer", "designer", "scientist", "manager", "analyst",
    "python", "java", "react", "node", "dotnet", ".net", "django", "spring",
    "experience", "skills", "projects", "education",
}

def _title_case_name(s: str) -> str:
    parts = [p for p in re.split(r"[\s\._\-]+", s) if p]
    parts = [p for p in parts if not re.fullmatch(r"\d+", p)]
    return " ".join(p.capitalize() for p in parts if p)


def _is_probable_name(line: str) -> bool:
    t = line.strip()
    if not (2 <= len(t.split()) <= 4):
        return False
    if any(ch.isdigit() for ch in t):
        return False
    low = t.lower()
    if any(tok in low for tok in _NAME_BAD_TOKENS):
        return False
    # Require each token to look like a name-ish token
    for w in t.split():
        if not re.match(r"^[A-Z][a-z'’\-]+$", w):
            return False
    return True


def extract_name(text: str) -> Optional[str]:
    # 1) NER-based (PERSON)
    if _HAS_NER:
        try:
            doc = nlp(text)
            # Prefer the earliest PERSON near top
            for ent in doc.ents:
                if getattr(ent, "label_", "") == "PERSON":
                    candidate = ent.text.strip()
                    if _is_probable_name(candidate):
                        return candidate
        except Exception:
            pass

    # 2) Top lines heuristic
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    for line in lines[:12]:
        # skip contact/links lines quickly
        if any(x in line.lower() for x in ["linkedin", "github", "email", "phone", "contact"]):
            continue
        if _is_probable_name(line):
            return line

    # 3) “Name: …” pattern
    match = re.search(r"(?im)^\s*name[:\s]+([A-Z][a-z'’\-]+(?:\s+[A-Z][a-z'’\-]+){1,3})\s*$", text)
    if match:
        return match.group(1).strip()

    # 4) Fallback via email username (e.g., john.doe -> John Doe)
    email = extract_email(text)
    if email:
        username = email.split("@", 1)[0]
        guess = _title_case_name(username)
        if guess and 2 <= len(guess.split()) <= 4:
            return guess

    return None


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group() if match else None


def extract_phone(text: str) -> Optional[str]:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group() if match else None


def _skills_from_explicit_section(text: str) -> List[str]:
    """
    Pull skills from an explicit "Skills" section, tolerant to variants.
    """
    try:
        # Capture content after a skills-like heading up to next major heading or blank gap
        sec = re.search(
            r"(?is)\b(skills|technical skills|core skills|skills summary)\b[:\n\-–]*"
            r"([\s\S]*?)(?=\n\s*[A-Z][A-Za-z ]{2,30}[:\n]|"
            r"\n\s*[A-Z ]{3,}\n|"
            r"\n\s*(?:education|experience|projects|certifications)\b|$)",
            text,
        )
        if not sec:
            return []
        body = sec.group(2) or ""
        # Split by common delimiters
        parts = re.split(r"[,\|;/•·●\n\r]+", body)
        skills: List[str] = []
        for p in parts:
            t = re.sub(r"[\(\)\[\]{}]", " ", p).strip()
            t = re.sub(r"\s+", " ", t)
            if not t:
                continue
            low = t.lower()
            # drop obvious noise words
            if low in STOPWORDS or len(low) < 2:
                continue
            # e.g., "Programming Languages: Python" -> keep after colon
            if ":" in low:
                low = low.split(":", 1)[-1].strip()
            # filter very long phrases that are not real skills
            if len(low) > 48:
                continue
            skills.append(low)
        # de-dup while preserving order
        out, seen = [], set()
        for s in skills:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    except Exception:
        return []


def extract_skills(text: str) -> List[str]:
    """
    Skills extraction (enhanced):
      1) Use explicit Skills section if present (broad tokenization)
      2) Phrase-level contains check from skills.txt (multi-word first)
      3) Token-level fallback when vocab present
    """
    low_all = text.lower()
    matched: set = set()

    # 1) Section-based parse
    section_skills = _skills_from_explicit_section(text)
    matched |= set(section_skills)

    # 2) Phrase contains (longest first) from vocab
    if KNOWN_SKILLS:
        for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
            try:
                if skill and skill in low_all:
                    matched.add(skill)
            except Exception:
                continue

    # 3) Token-level fallback via spaCy vocab overlap
    try:
        doc = nlp(low_all)
        tokens = set(
            getattr(t, "text", "") for t in doc
            if getattr(t, "is_stop", False) is False and getattr(t, "is_punct", False) is False
        )
        matched |= (KNOWN_SKILLS.intersection(tokens))
    except Exception:
        pass

    # return normalized, de-duplicated list
    out = []
    seen = set()
    for m in sorted({m.strip().lower() for m in matched if m and m.strip()}):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


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
        for m in (year_match or []):
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
            r'(\w+\s\d{4})\s*(?:-|–|—|to)\s*(\w+\s\d{4}|present|current)',
            text,
            re.IGNORECASE
        )
        total_months = 0
        for start, end in date_ranges:
            try:
                start_date = dateparser.parse(start)
                end_date = datetime.now() if re.search(r"(present|current)", end, re.IGNORECASE) else dateparser.parse(end)
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
    ✅ Fixed: Extract projects ONLY from project-specific sections.
    Avoids mixing with location, education, or other categories.
    """
    projects: List[Dict[str, Any]] = []
    seen = set()
    
    # ✅ 1) Look for explicit "Projects" section first (most reliable)
    project_section_patterns = [
        r'(?is)\b(?:projects?|portfolio|key projects?|notable projects?)\b[:\n\-–]*([\s\S]{0,2000})',
    ]
    
    for pattern in project_section_patterns:
        try:
            matches = re.finditer(pattern, text)
            for match in matches:
                section_text = match.group(1).strip()
                if not section_text or len(section_text) < 20:
                    continue
                
                # Split by bullet points or numbered items
                items = re.split(r'\n\s*[-•*]\s+|\n\s*\d+[\.\)]\s+', section_text)
                for item in items:
                    item = item.strip()
                    if len(item) < 20:  # Too short to be a project
                        continue
                    
                    # Extract project name (first line or first sentence)
                    lines = [l.strip() for l in item.split('\n') if l.strip()]
                    if not lines:
                        continue
                    
                    # First line is usually project name
                    name = lines[0][:100].strip()
                    # Rest is description
                    description = " ".join(lines[1:]).strip() or lines[0][100:].strip()
                    
                    # ✅ Validate: exclude location-like patterns
                    name_lower = name.lower()
                    if any(word in name_lower for word in ["city", "country", "state", "province", "region"]):
                        continue
                    
                    # ✅ Validate: must contain project-like keywords or tech terms
                    if not re.search(r'\b(?:project|application|system|platform|website|app|software|tool|framework|api|service)\b', item, re.I):
                        # If no project keywords, check for tech skills (indicates it's a tech project)
                        if not KNOWN_SKILLS or not any(skill in item.lower() for skill in KNOWN_SKILLS[:20]):
                            continue
                    
                    if name and name not in seen:
                        seen.add(name)
                        
                        # Extract technologies
                        tech: List[str] = []
                        item_lower = item.lower()
                        if KNOWN_SKILLS:
                            for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
                                if skill and skill in item_lower:
                                    tech.append(skill)
                        
                        projects.append({
                            "name": name,
                            "description": description[:500],  # Limit description length
                            "tech": sorted(list(set(tech)))
                        })
                        
                        if len(projects) >= 5:
                            break
                if len(projects) >= 5:
                    break
        except Exception:
            continue
    
    # ✅ 2) Fallback: Look for work history entries that mention projects
    if len(projects) < 3:
        try:
            work_pattern = r'(?is)\b(?:experience|work history|employment)\b[:\n\-–]*([\s\S]{0,3000})'
            work_match = re.search(work_pattern, text)
            if work_match:
                work_text = work_match.group(1)
                # Look for project mentions within work history
                project_mentions = re.finditer(
                    r'(?i)\b(?:project|developed|built|created|designed|implemented)\b[:\s]+([^\n]{20,200})',
                    work_text
                )
                for mention in project_mentions:
                    desc = mention.group(1).strip()
                    if len(desc) < 20:
                        continue
                    # Extract as project name
                    name = desc[:80]
                    if name not in seen:
                        seen.add(name)
                        tech: List[str] = []
                        desc_lower = desc.lower()
                        if KNOWN_SKILLS:
                            for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
                                if skill and skill in desc_lower:
                                    tech.append(skill)
                        projects.append({
                            "name": name,
                            "description": desc,
                            "tech": sorted(list(set(tech)))
                        })
                        if len(projects) >= 5:
                            break
        except Exception:
            pass

    return projects[:5]  # Return max 5 projects


def extract_summary(text: str) -> str:
    try:
        match = re.search(
            r"(?is)\b(Summary|Objective|Profile|Professional Summary)\b[:\n\s]*(.*?)(?:\n\n|\n[A-Z][a-z]+:|\n[A-Z ]{3,}|Education|Experience|Projects|Skills|$)",
            text
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
    """
    Improved education parsing:
      - Detect education section
      - Extract degree (normalized), school, year, GPA if available
    """
    results = []

    # Narrow to education section when possible
    try:
        sec = re.search(
            r"(?is)\b(education|academic qualifications|academics|qualifications)\b[:\n\-–]*"
            r"([\s\S]*?)(?=\n\s*[A-Z][A-Za-z ]{2,30}[:\n]|"
            r"\n\s*[A-Z ]{3,}\n|"
            r"\n\s*(?:experience|projects|skills|certifications)\b|$)",
            text,
        )
        body = sec.group(2) if sec else text
    except Exception:
        body = text

    # Split into moderate lines and attempt to parse degree/school/year/gpa in each
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or len(line) < 3:
            continue

        # Pull year (4 digits) if present
        year_match = re.search(r"(19|20)\d{2}", line)
        year = year_match.group(0) if year_match else None

        # GPA / CGPA (e.g., 8.2/10 or 3.7 or 3.70)
        gpa = None
        gpa_m = re.search(r"(?i)(?:cgpa|gpa)\s*[:\-]?\s*([0-9]+(?:\.\d{1,2})?(?:\s*/\s*[0-9]+(?:\.\d{1,2})?)?)", line)
        if gpa_m:
            gpa = gpa_m.group(1).strip()

        # Degree heuristic (normalize via normalize_degree)
        deg_m = re.search(
            r"(?i)(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc|b\.?a|m\.?a|b\.?com|bcom|bca|mca|mba|pgdm|ph\.?d|ll\.?b|ll\.?m|bachelor of [a-z ]+|master of [a-z ]+)",
            line
        )
        degree_raw = deg_m.group(0) if deg_m else None
        degree = normalize_degree(degree_raw) if degree_raw else None

        # School/University heuristic
        school = None
        sch_m = re.search(r"(?i)\b([A-Za-z][A-Za-z&\.\- ]{2,80})\b(?:,?\s*(?:university|college|institute|school))", line)
        if sch_m:
            school = sch_m.group(0).strip()
        else:
            # Sometimes school precedes degree
            sch_m2 = re.search(r"(?i)\b([A-Za-z][A-Za-z&\.\- ]{4,80})(?:,|\s)-?\s*(?:\d{4}|\b)", line)
            if sch_m2 and not degree:
                school = sch_m2.group(1).strip()

        if degree or school or year or gpa:
            results.append({
                "degree": degree or (degree_raw.strip() if degree_raw else None) or "",
                "school": (school or "").strip(),
                "year": year or "",
                "gpa": gpa or ""
            })

    return results


def _extract_experience_blocks(text: str) -> list:
    """
    Heuristic extractor for work history entries using common patterns:
      - Title @ Company (Dates)
      - Company - Title  Dates
      - Title, Company  (Month YYYY - Present)
    Returns list of dicts with title, company, duration, description.
    """
    results = []
    # Standard month patterns
    MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?" \
            r"|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    DATE = rf"(?:{MONTH}\s+\d{{4}})"
    DUR = rf"(?:{DATE}|Present|Current)"
    DURRANGE = rf"{DUR}\s*(?:-|–|—|to)\s*{DUR}"

    # Pattern 1: Title @ Company (Duration)
    pat1 = re.compile(
        rf"(?im)^\s*(?P<title>[\w&/.,'’\-\s]{{3,80}}?)\s*(?:@|-|,)\s*"
        rf"(?P<company>[\w&/.,'’\-\s]{{3,80}}?)\s*(?:\(|-|\s)\s*(?P<duration>{DURRANGE})?\)?\s*$"
    )

    # Pattern 2: Company - Title (Duration)
    pat2 = re.compile(
        rf"(?im)^\s*(?P<company>[\w&/.,'’\-\s]{{3,80}}?)\s*(?:-|,|@)\s*"
        rf"(?P<title>[\w&/.,'’\-\s]{{3,80}}?)\s*(?:\(|-|\s)\s*(?P<duration>{DURRANGE})?\)?\s*$"
    )

    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    for i, ln in enumerate(lines):
        m = pat1.match(ln) or pat2.match(ln)
        if not m:
            continue
        title = m.group("title").strip()
        company = m.group("company").strip()
        duration = (m.group("duration") or "").strip()

        # Attempt to capture a short description from following bullet lines
        desc_lines = []
        for j in range(i + 1, min(i + 6, n)):
            t = lines[j].strip()
            if re.match(r"^\s*(?:[-*•·●]|[0-9]+\.)\s+", lines[j]):
                desc_lines.append(re.sub(r"^\s*(?:[-*•·●]|[0-9]+\.)\s+", "", t))
            else:
                # stop if next clear heading
                if re.match(r"^[A-Z][A-Za-z ]{2,30}:?$", t) or len(t.split()) <= 2:
                    break
                # allow one plain line if reasonably long
                if len(t) > 30:
                    desc_lines.append(t)
                else:
                    break

        results.append({
            "title": norm_text(title) or title,
            "company": norm_text(company) or company,
            "duration": duration,
            "description": " ".join(desc_lines)[:400] if desc_lines else "Worked on various tasks and contributed to team goals."
        })

    return results


def extract_work_history(text: str) -> list:
    """
    Enhanced work history extraction that first attempts structured patterns,
    then falls back to the original simple 3-line grouping.
    """
    # 1) Try structured patterns
    items = _extract_experience_blocks(text)
    if items:
        return items

    # 2) Fallback to the prior heuristic (kept for backward compatibility)
    try:
        jobs = re.findall(r"(?is)\b(Experience|Work Experience|Employment History|Work)\b[\s\S]{0,1000}", text, re.IGNORECASE)
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
                    "title": norm_text(title) or title,
                    "company": norm_text(company) or company,
                    "duration": duration,
                    "description": "Worked on various tasks and contributed to team goals."
                })
    return results


def _split_city_country(loc: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Basic split of "City, Country" → ("City", "Country").
    """
    try:
        parts = [x.strip() for x in re.split(r"[,/|-]", loc) if x.strip()]
        if len(parts) >= 2:
            # heuristic: first looks like city if it's 1-3 words
            city = parts[0]
            country = parts[-1]
            return city, country
    except Exception:
        pass
    return None, None


def extract_location(text: str) -> str:
    """
    ✅ Fixed: Extract location ONLY from location-specific fields.
    Avoids mixing with projects, roles, or other categories.
    """
    # Common location patterns to exclude (not actual locations)
    non_location_patterns = [
        r"curriculum", r"classroom", r"design", r"development", r"project",
        r"teaching", r"education", r"training", r"course", r"lesson"
    ]
    
    # 1) spaCy NER (GPE) - most reliable
    if _HAS_NER:
        try:
            doc = nlp(text)
            locations = []
            for ent in doc.ents:
                label = getattr(ent, "label_", "")
                if label == "GPE":  # Geopolitical entity (countries, cities, states)
                    loc_text = ent.text.strip()
                    # ✅ Filter out non-location patterns
                    loc_lower = loc_text.lower()
                    if not any(re.search(pat, loc_lower) for pat in non_location_patterns):
                        locations.append(loc_text)
            if locations:
                # Prefer longer location strings (more specific)
                best = max(locations, key=len)
                normalized = normalize_location(best)
                if normalized and normalized != "N/A":
                    return normalized
                return best
        except Exception:
            pass

    # 2) Explicit location section patterns (more reliable)
    location_section_patterns = [
        r"(?im)\b(?:location|address|based in|located in|residing in|current location)\b[: ]+\s*([A-Z][a-zA-Z\s,]+?)(?:\n|$|,|;|\.)",
        r"(?im)\b(?:city|country)\b[: ]+\s*([A-Z][a-zA-Z\s,]+?)(?:\n|$|,|;|\.)",
    ]
    for pattern in location_section_patterns:
        m = re.search(pattern, text)
        if m:
            cand = m.group(1).strip()
            # ✅ Validate: must look like a location (city/country names, not project names)
            cand_lower = cand.lower()
            if not any(re.search(pat, cand_lower) for pat in non_location_patterns):
                # Check if it contains common location indicators
                if re.search(r"\b(city|country|state|province|region|area)\b", cand_lower, re.I):
                    normalized = normalize_location(cand)
                    if normalized and normalized != "N/A":
                        return normalized
                # Or if it's a known city/country pattern
                elif re.match(r"^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?(?:,\s*[A-Z][a-zA-Z]+)?$", cand):
                    normalized = normalize_location(cand)
                    if normalized and normalized != "N/A":
                        return normalized
                    return cand

    # 3) Email signature style "City, Country" near contact info
    # Only match if near email/phone patterns
    contact_context = re.search(r"(?:email|phone|contact|@)[\s\S]{0,200}([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?),\s*([A-Z][a-zA-Z]+)", text, re.IGNORECASE)
    if contact_context:
        city = contact_context.group(1).strip()
        country = contact_context.group(2).strip()
        cand = f"{city}, {country}"
        # ✅ Validate: exclude non-location patterns
        cand_lower = cand.lower()
        if not any(re.search(pat, cand_lower) for pat in non_location_patterns):
            normalized = normalize_location(cand)
            if normalized and normalized != "N/A":
                return normalized
            return cand

    return "N/A"

# ----------------------------
# Role/category classification (deterministic, confidence-backed)
# ----------------------------
_WORD_RE = re.compile(r"\b[a-z0-9][a-z0-9 +./_-]*\b")

def _count_occurrences(blob: str, term: str) -> int:
    """
    Count occurrences of term in blob with word-boundary-ish regex.
    Spaces in term are honored. Case-insensitive, blob expected lowercased.
    """
    term = term.strip().lower()
    if not term:
        return 0
    # escape + keep wildcards simple
    pat = re.escape(term)
    # allow separators between words (tolerant)
    pat = pat.replace(r"\ ", r"\s+")
    try:
        return len(re.findall(rf"(?<![a-z0-9]){pat}(?![a-z0-9])", blob, flags=re.IGNORECASE))
    except Exception:
        return blob.count(term)

def _score_by_keywords(blob: str, skills_norm: List[str], terms: List[str]) -> float:
    """
    Score a label by summing term occurrences. Phrases weighed higher.
    Bonus if a term is also present in skills_norm.
    """
    score = 0.0
    skills_set = set(skills_norm or [])
    for t in terms or []:
        t = (t or "").strip().lower()
        if not t:
            continue
        occ = _count_occurrences(blob, t)
        if occ <= 0:
            continue
        weight = 2.0 if " " in t else 1.0
        score += occ * weight
        # small bonus for explicit skill presence
        if t in skills_set:
            score += 0.5
    return score

def _pick_best_label(blob: str, skills_norm: List[str], mapping: Dict[str, List[str]]) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Return (best_label, confidence, scores_dict). Confidence is stable (0-1),
    based on margin between top-2 scores.
    """
    scores: Dict[str, float] = {}
    for label, terms in mapping.items():
        s = _score_by_keywords(blob, skills_norm, terms or [])
        scores[label] = s

    # pick best
    best_label = None
    best_score = 0.0
    second = 0.0
    for k, v in scores.items():
        if v > best_score:
            second = best_score
            best_score = v
            best_label = k
        elif v > second:
            second = v

    if not best_label or best_score <= 0:
        return None, 0.0, scores

    # margin-based confidence, deterministic
    margin = best_score - second
    # normalized margin ratio ∈ [0,1]
    ratio = margin / (best_score if best_score > 0 else 1.0)
    confidence = 0.6 + 0.4 * max(0.0, min(1.0, ratio))  # ∈ [0.6, 1.0]
    return best_label, round(confidence, 3), scores

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
    experience_display = _format_experience_display(experience_float)  # clean display
    experience_rounded = _round_experience_value(experience_float)     # numeric rounded

    # Additive detections (for strong filtering convenience)
    degrees_detected = _detect_degrees(text or "")
    schools_detected = _detect_schools(text or "")

    # ---------- Precompute search_blob early (used by classification) ----------
    try:
        parts: List[str] = []
        parts.append(name or "")
        parts.append(current_title or "")
        parts.append(location or "")
        if skills:
            parts.append(" ".join([s for s in skills if isinstance(s, str)]))
        if projects:
            proj_bits: List[str] = []
            for p in projects[:6]:
                if isinstance(p, dict):
                    proj_bits.append(str(p.get("name") or ""))
                    proj_bits.append(str(p.get("description") or ""))
                elif isinstance(p, str):
                    proj_bits.append(p)
            parts.append(" ".join([x for x in proj_bits if x]))
        if summary:
            parts.append(summary[:800])
        # include raw (truncated)
        raw = text or ""
        truncate_chars = int(os.getenv("SEARCH_BLOB_TRUNCATE_CHARS", "4000"))
        if raw:
            parts.append(raw[:truncate_chars])
        # Build blob using shared helper (keeps behavior aligned)
        search_blob = make_search_blob([p for p in parts if p], truncate_chars=truncate_chars)
        if not search_blob:
            # fallback to local cleaner just in case
            search_blob = _clean_text_for_blob(" ".join([p for p in parts if p]).strip())
    except Exception:
        search_blob = _clean_text_for_blob((text or "")[:4000])

    # ---------- Normalized canonical fields ----------
    role_source = (current_title or "").strip() or None
    role_norm = normalize_role(role_source)
    skills_norm = normalize_tokens(skills)
    proj_items: List[str] = []
    for p in projects:
        if isinstance(p, dict):
            if p.get("name"):
                proj_items.append(str(p["name"]))
            if isinstance(p.get("tech"), list):
                proj_items.extend([str(t) for t in p["tech"] if isinstance(t, (str,))])
    projects_norm = normalize_tokens(proj_items)
    yoe_num = to_years_of_experience(experience_float)
    location_norm = norm_text(location) or None
    city, country = _split_city_country(location) if location and location != "N/A" else (None, None)

    # ---------- Deterministic role/category prediction with confidence ----------
    # Use search_blob (lowercased) + skills_norm for scoring
    best_role, role_conf, _role_scores = _pick_best_label(search_blob, skills_norm, ROLE_KEYWORDS)
    best_cat, cat_conf, _cat_scores = _pick_best_label(search_blob, skills_norm, CATEGORY_KEYWORDS)

    # fallbacks: if classifier couldn't find, use normalized current title as role
    predicted_role = best_role or (role_norm if role_norm else None)

    info = {
        "name": name or "No Name",                 # fallback
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

        # ✅ Title/Role: Store only canonical fields
        "currentRole": current_title or "N/A",     # Canonical field
        "title": current_title or "N/A",           # Alias (for backward compatibility)

        # Projects
        "projects": projects,
        "project_summary": project_summary_text,
        "project_details": project_summary_text,
        "portfolio": project_summary_text,

        # ✅ Location: Store only canonical fields
        "location": location,                      # Canonical field (full location string)
        "location_norm": location_norm,           # Normalized for filtering
        # city and country can be derived from location if needed, but store for performance
        "city": city,                              # Extracted city (if available)
        "country": country,                        # Extracted country (if available)

        # Raw/Index text
        "raw_text": text,
        "search_blob": search_blob,
        "index_blob": search_blob,
        "index_embedding": None,  # to be filled by upstream ML if enabled

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

        # ---------- Normalized canonical fields for fast filtering ----------
        "role_norm": role_norm,
        "skills_norm": skills_norm,
        "projects_norm": projects_norm,
        "yoe_num": yoe_num,

        # ---------- Role & Category predictions ----------
        "predicted_role": predicted_role or None,
        "ml_confidence": role_conf if predicted_role else 0.0,     # for cards: Role Prediction Confidence
        "role_confidence": role_conf if predicted_role else 0.0,   # alias
        "category": best_cat or None,
        "category_confidence": cat_conf if best_cat else 0.0,
    }

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
