"""
Unit tests for Calibrated Judge and Cohen's Kappa calculation.
"""
from src.calibrated_judge import CalibratedJudge, calculate_cohens_kappa


def test_calibrated_judge_pass_and_fail():
    judge = CalibratedJudge(faithfulness_threshold=0.70)
    context = "Exit Code 137 indicates SIGKILL by Linux OOM killer."

    # Answerable & Faithful -> PASS
    v_pass = judge.evaluate_response(
        prompt="What is exit code 137?",
        context=context,
        candidate_answer="Exit Code 137 indicates SIGKILL by OOM killer.",
        is_answerable=True
    )
    assert v_pass.verdict == "PASS"

    # Answerable & Unfaithful -> FAIL
    v_fail = judge.evaluate_response(
        prompt="What is exit code 137?",
        context=context,
        candidate_answer="Exit Code 137 indicates a missing configuration YAML file.",
        is_answerable=True
    )
    assert v_fail.verdict == "FAIL"

    # Unanswerable & Properly Refused -> PASS
    v_refuse_ok = judge.evaluate_response(
        prompt="How to bake cake?",
        context="Systems manual",
        candidate_answer="I cannot answer this question based on the provided documentation.",
        is_answerable=False
    )
    assert v_refuse_ok.verdict == "PASS"


def test_cohens_kappa():
    rater1 = [1, 1, 1, 0, 0, 1, 0, 1]
    rater2 = [1, 1, 1, 0, 0, 1, 0, 1]  # Perfect agreement
    kappa_perfect = calculate_cohens_kappa(rater1, rater2)
    assert kappa_perfect == 1.0

    rater3 = [1, 1, 1, 0, 0, 1, 1, 1]  # High agreement
    kappa_high = calculate_cohens_kappa(rater1, rater3)
    assert kappa_high >= 0.70
