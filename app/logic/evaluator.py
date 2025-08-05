def evaluate_mcq(submitted, correct):
    """
    Evaluates a single MCQ answer.
    """
    return 1 if submitted.strip().lower() == correct.strip().lower() else 0

def evaluate_test(answers, correct_answers):
    """
    Evaluates a test by comparing submitted answers with correct ones.
    Returns a total score and detailed breakdown.
    """
    score = 0
    detailed_results = []

    for i, submitted in enumerate(answers):
        if i >= len(correct_answers):
            continue  # safety check

        correct = correct_answers[i]
        submitted_ans = submitted.get("answer", "").strip().lower()
        correct_ans = correct.get("correct_answer", "").strip().lower()
        q_type = correct.get("type", "mcq")

        is_correct = False
        explanation = ""

        if q_type == "mcq":
            is_correct = submitted_ans == correct_ans
            score += int(is_correct)
            explanation = "Correct answer matched." if is_correct else f"Expected '{correct_ans}', got '{submitted_ans}'"

        detailed_results.append({
            "question": correct.get("question", ""),
            "submitted": submitted_ans,
            "correct": correct_ans,
            "is_correct": is_correct,
            "explanation": explanation
        })

    return {
        "total_score": score,
        "details": detailed_results
    }
