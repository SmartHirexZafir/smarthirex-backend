# ✅ File: app/logic/resume_parser.py

import fitz  # PyMuPDF
import docx2txt
import io
import re
from datetime import datetime
from dateutil import parser as dateparser

# Define a broader skillset list (can be extended)
SKILL_KEYWORDS = [
    "python", "java", "sql", "react", "node", "html", "css",
    "docker", "aws", "ml", "ai", "js", "typescript", "c++",
    "c#", "angular", "flask", "django", "mongodb", "excel",
    "power bi", "tableau", "selenium", "git", "kubernetes"
]

def extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(content: bytes) -> str:
    with io.BytesIO(content) as f:
        return docx2txt.process(f)

def extract_info(text: str) -> dict:
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)
    experience = extract_experience(text)
    projects = extract_projects(text)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience": experience,
        "projects": projects,
        "raw_text": text
    }

def extract_name(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines[:5]:  # check top 5 lines
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

def extract_skills(text: str) -> list:
    found = []
    lower_text = text.lower()
    for skill in SKILL_KEYWORDS:
        if skill in lower_text:
            found.append(skill)
    return list(set(found))

def extract_experience(text: str) -> int:
    # Match phrases like "5+ years of experience", "3 years"
    year_match = re.findall(r"(\d+)[+\s]*(?:years|yrs)[\s\w]*experience", text, re.IGNORECASE)
    max_years = max(map(int, year_match)) if year_match else 0

    # Also check date ranges like "Jan 2018 - Mar 2022"
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

def parse_resume_file(filename: str, content: bytes) -> dict:
    text = extract_text_from_pdf(content) if filename.endswith(".pdf") else extract_text_from_docx(content)
    return extract_info(text)
