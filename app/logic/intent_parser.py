# ✅ File: app/logic/intent_parser.py

import re

def detect_usage_help(prompt: str) -> bool:
    """
    Detects if the user is asking for help/about information.
    """
    keywords = [
        "kaise", "kahan", "use", "signup", "interview", "shortlist",
        "start", "register", "filter dikh", "candidate", "how to", "where"
    ]
    return any(word in prompt.lower() for word in keywords)

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

    # Experience extraction (e.g., "5 years", "3+ yrs")
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
