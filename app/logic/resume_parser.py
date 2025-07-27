# ✅ File: app/logic/resume_parser.py

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

def extract_name(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines[:5]:
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
    return list(sorted(matched))

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

def extract_projects(text: str) -> list:
    project_sections = re.findall(r'(?:Project[s]?:?|Responsibilities:|Description:)[\s\S]{0,1000}', text, re.IGNORECASE)
    projects = []
    for section in project_sections:
        lines = section.strip().split("\n")
        filtered = [line.strip() for line in lines if len(line.strip()) > 20]
        if filtered:
            projects.append(" ".join(filtered[:3]))
    return projects

def extract_summary(text: str) -> str:
    summary_match = re.search(r"(?:Summary|Objective)[:\n\s]*(.*?)(?:Education|Experience|Projects|Skills|$)", text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip().split('\n')[0]
        return summary[:500]
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
    jobs = re.findall(r"(?:Experience|Work)[\s\S]{0,800}", text, re.IGNORECASE)
    results = []
    for job in jobs:
        lines = job.split('\n')
        for i in range(0, len(lines)-2, 3):
            title = lines[i].strip()
            company = lines[i+1].strip()
            duration = lines[i+2].strip()
            if len(title) > 2 and len(company) > 2 and len(duration) > 2:
                results.append({
                    "title": title,
                    "company": company,
                    "duration": duration,
                    "description": "Worked on various tasks and contributed to team goals."
                })
    return results

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

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience": experience,
        "projects": projects,
        "raw_text": text,

        "resume": {
            "summary": summary,
            "education": education,
            "workHistory": workHistory,
            "projects": projects,
        },
        "currentRole": workHistory[0]["title"] if workHistory else "N/A",
        "company": workHistory[0]["company"] if workHistory else "N/A",
        "location": "N/A"
    }

def parse_resume_file(filename: str, content: bytes) -> dict:
    text = extract_text_from_pdf(content) if filename.endswith(".pdf") else extract_text_from_docx(content)
    return extract_info(text)
