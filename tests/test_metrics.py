"""
Unit tests for evaluation metrics.
"""
from src.metrics import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    is_refusal,
    compute_hallucination_rate
)


def test_faithfulness_and_hallucination():
    context = "Raft uses randomized election timeouts between 150ms and 300ms."
    good_answer = "Election timeouts in Raft are randomized between 150ms and 300ms."
    bad_answer = "Raft fixed election timeouts at 1000ms using Paxos rounds."

    faith_good = evaluate_faithfulness(good_answer, context)
    faith_bad = evaluate_faithfulness(bad_answer, context)

    assert faith_good >= 0.80
    assert faith_bad < 0.50
    assert compute_hallucination_rate(faith_good) <= 0.20
    assert compute_hallucination_rate(faith_bad) >= 0.50


def test_refusal_detection():
    refusal_text = "I cannot answer this question based on the provided documentation."
    normal_text = "Kubernetes pods restart automatically under CrashLoopBackOff."

    assert is_refusal(refusal_text)
    assert not is_refusal(normal_text)
    assert evaluate_faithfulness(refusal_text, "Some unrelated context") == 1.0


def test_answer_relevance():
    prompt = "What causes container Exit Code 137 in Kubernetes?"
    relevant_answer = "Exit Code 137 is caused by the Linux OOM killer."
    irrelevant_answer = "The weather today in San Francisco is sunny."

    rel_high = evaluate_answer_relevance(prompt, relevant_answer)
    rel_low = evaluate_answer_relevance(prompt, irrelevant_answer)

    assert rel_high > rel_low
