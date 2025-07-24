# app/logic/evaluator.py

def evaluate_mcq(submitted, correct):
    """
    Evaluates a single MCQ answer.
    """
    return 1 if submitted.strip().lower() == correct.strip().lower() else 0

def evaluate_test(answers, correct_answers):
    """
    Evaluates a test by comparing submitted answers with correct ones.
    Returns a total score.
    """
    score = 0
    for i, submitted in enumerate(answers):
        if i >= len(correct_answers):
            continue  # safety check if lengths mismatch

        correct = correct_answers[i]
        if correct["type"] == "mcq":
            score += evaluate_mcq(submitted["answer"], correct["correct_answer"])

    return score
