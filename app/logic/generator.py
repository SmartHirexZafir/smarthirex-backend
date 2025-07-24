# ✅ File: app/logic/generator.py
from app.models.auto_test_models import Candidate
import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_test(candidate: Candidate):
    level = "junior" if int(candidate.experience.split()[0]) < 3 else "senior"

    prompt = f"""
    You are an HR test generator bot. Create a {level}-level test for a candidate:
    - Experience: {candidate.experience}
    - Skills: {', '.join(candidate.skills)}
    - Job Role: {candidate.job_role}

    Include:
    - 2 MCQs
    - 1 Coding question
    - 1 Scenario-based question

    Format your response strictly as valid JSON:
    [
      {{ "type": "mcq", "question": "...", "options": ["A", "B", "C", "D"], "correct_answer": "B" }},
      {{ "type": "code", "question": "...", "correct_answer": null }},
      ...
    ]
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        return [{"type": "mcq", "question": "Fallback MCQ", "options": ["A", "B", "C", "D"], "correct_answer": "A"}]
