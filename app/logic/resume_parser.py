import fitz  # PyMuPDF
import docx2txt
import io
import re
from datetime import datetime
from dateutil import parser as dateparser
import spacy
from pathlib import Path

# ✅ Load SpaCy NLP model once
nlp = spacy.load("en_core_web_sm")

# ✅ Load known skills from external file
SKILL_FILE = Path(__file__).parent.parent / "resources" / "skills.txt"
KNOWN_SKILLS = set()
if SKILL_FILE.exists():
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        KNOWN_SKILLS = set(line.strip().lower() for line in f if line.strip())

def extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(content: bytes) -> str:
    with io.BytesIO(content) as f:
        return docx2txt.process(f)

# ✅ Name extractor: NLP entity + regex fallback
def extract_name(text: str) -> str:
    doc = nlp(text)
    names = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and 2 <= len(ent.text.strip().split()) <= 4]
    if names:
        return names[0]

    # Fallback to regex
    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+$", line):
            return line
    match = re.search(r"(?i)^Name[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)+)", text)
    return match.group(1) if match else None

def extract_email(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group() if match else None

def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group() if match else None

# ✅ NLP-based skill extractor using skills.txt
def extract_skills(text: str) -> list:
    doc = nlp(text.lower())
    tokens = set(token.text for token in doc if not token.is_stop and not token.is_punct)
    matched = KNOWN_SKILLS.intersection(tokens)
    return sorted(list(matched))

def extract_experience(text: str) -> int:
    year_match = re.findall(r"(\d+)[+\s]*(?:years|yrs)[\s\w]*experience", text, re.IGNORECASE)
    max_years = max(map(int, year_match)) if year_match else 0

    date_ranges = re.findall(r'(\w+\s\d{4})\s*[-–to]+\s*(\w+\s\d{4}|present)', text, re.IGNORECASE)
    total_months = 0
    for start, end in date_ranges:
        try:
            start_date = dateparser.parse(start)
            end_date = datetime.now() if "present" in end.lower() else dateparser.parse(end)
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += max(months, 0)
        except:
            continue
    calculated_years = round(total_months / 12)
    return max(max_years, calculated_years)

# ✅ Structured project extractor
def extract_projects(text: str) -> list:
    raw_sections = re.findall(r'(?:Project[s]?:?|Responsibilities:|Description:)[\s\S]{0,800}', text, re.IGNORECASE)
    projects = []
    seen = set()
    for section in raw_sections:
        lines = [line.strip() for line in section.strip().split('\n') if len(line.strip()) > 20]
        if lines:
            name = lines[0][:80]
            description = " ".join(lines[1:]).strip()
            if name not in seen:
                seen.add(name)
                projects.append({
                    "name": name,
                    "description": description,
                    "tech": []
                })
    return projects[:5]  # Limit to top 5 projects

# ✅ Full summary extractor with NLP fallback
def extract_summary(text: str) -> str:
    match = re.search(
        r"(Summary|Objective)[:\n\s]*(.*?)(?:\n\n|\n[A-Z][a-z]+:|\n[A-Z ]{3,}|Education|Experience|Projects|Skills|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(2).strip()[:600]

    # NLP fallback: find first paragraph with 2+ verbs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) > 10]
    for para in paragraphs:
        doc = nlp(para)
        verbs = [token for token in doc if token.pos_ == "VERB"]
        if len(verbs) >= 2:
            return para.strip()[:600]
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

# ✅ NLP + fallback for location
def extract_location(text: str) -> str:
    doc = nlp(text)
    locations = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]
    if locations:
        return locations[0]
    match = re.search(r"\b(?:located in|from|based in)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text)
    return match.group(1) if match else "N/A"

# ✅ Master extractor — resume → structured dict
def extract_info(text: str) -> dict:
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)
    experience = extract_experience(text)
    projects = extract_projects(text)
    summary = extract_summary(text)
    education = extract_education(text)
    workHistory = extract_work_history(text)
    location = extract_location(text)

    return {
        "name": name or "Unnamed Candidate",
        "email": email or None,
        "phone": phone or None,
        "skills": skills or [],
        "experience": experience,
        "projects": projects,
        "raw_text": text,

        "resume": {
            "summary": summary,
            "education": education,
            "workHistory": workHistory,
            "projects": projects
        },
        "currentRole": workHistory[0]["title"] if workHistory else "N/A",
        "company": workHistory[0]["company"] if workHistory else "N/A",
        "location": location
    }

def parse_resume_file(filename: str, content: bytes) -> dict:
    text = extract_text_from_pdf(content) if filename.endswith(".pdf") else extract_text_from_docx(content)
    return extract_info(text)
