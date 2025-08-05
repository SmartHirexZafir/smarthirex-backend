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

def detect_show_all(prompt: str) -> bool:
    """
    Detects if the user wants to see all resumes, without any filter.
    """
    prompt = prompt.strip().lower()
    return prompt in [
        "show all",
        "list all",
        "show all candidates",
        "list all resumes",
        "display all",
        "get all",
        "sab dikhao",
        "saare candidate"
    ]

def parse_prompt(prompt: str) -> dict:
    """
    Parses the user's prompt to detect intent.
    """
    prompt = prompt.strip().lower()

    if detect_usage_help(prompt):
        return {
            "intent": "usage_help",
            "query": prompt
        }

    if detect_show_all(prompt):
        return {
            "intent": "show_all",
            "query": prompt
        }

    # Default: semantic CV filtering
    return {
        "intent": "filter_cv",
        "query": prompt
    }
