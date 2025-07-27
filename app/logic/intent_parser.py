# ✅ File: app/logic/intent_parser.py

import re

def detect_usage_help(prompt: str) -> bool:
    """
    Detects if the user is asking for help/about information.
    Only triggers if user is clearly asking 'how to use' the system.
    """
    prompt = prompt.lower()
    help_patterns = [
        r"\bhow to\b",
        r"\bhelp\b",
        r"\bkaise\b",
        r"\buse\b",
        r"\bstart\b",
        r"\bregister\b",
        r"\bsign ?up\b",
        r"\bguide\b"
    ]
    return any(re.search(pattern, prompt) for pattern in help_patterns)

def parse_prompt(prompt: str) -> dict:
    """
    Parses the user's prompt to extract intent, skills, experience,
    project keywords, and job role (if any).
    """
    prompt = prompt.lower()

    if detect_usage_help(prompt):
        return {"intent": "usage_help", "query": prompt}

    # Default intent
    intent = "filter_cv"

    # Skill keyword extraction
    known_skills = [
        "python", "java", "react", "node", "django", "sql", "aws",
        "angular", "ml", "ai", "html", "css", "javascript", "flask", "express"
    ]
    skills = list(set([s for s in known_skills if s in prompt]))

    # Experience extraction (e.g., "5 years", "3+ yrs", "5 saal")
    exp_match = re.search(r"(\d+)\+?\s*(?:years|yrs|saal)?", prompt)
    experience = exp_match.group(1) if exp_match else None

    # Project keyword extraction — catch phrases like "AI project", "ecommerce", "chatbot"
    project_keywords = re.findall(r"(?:project|kaam|worked on)\s+(?:in|with|on)?\s*([a-zA-Z0-9\-]+)", prompt)
    if "ai" in prompt and "ai" not in project_keywords:
        project_keywords.append("ai")
    if "chatbot" in prompt and "chatbot" not in project_keywords:
        project_keywords.append("chatbot")

    # Job role extraction — loose detection from common dev roles
    role_keywords = [
        "developer", "engineer", "data scientist", "web developer",
        "backend", "frontend", "full stack", "devops", "intern"
    ]
    job_role = None
    for phrase in role_keywords:
        if phrase in prompt:
            job_role = phrase
            break

    return {
        "intent": intent,
        "skills": skills,
        "experience": experience,
        "project_keywords": project_keywords,
        "job_role": job_role
    }
