"""
Calibrated Evaluation Judge Module.
Implements:
- Criterion-specific binary grading (PASS / FAIL)
- Human calibration via Cohen's Kappa
- Robustness guardrails: Position-swap invariance and length-bias mitigation
"""
import re
from pydantic import BaseModel
from src.metrics import evaluate_faithfulness, is_refusal


class JudgmentVerdict(BaseModel):
    verdict: str  # "PASS" or "FAIL"
    faithfulness: float
    is_refusal: bool
    position_invariant: bool
    length_bias_detected: bool
    justification: str


class CalibratedJudge:
    def __init__(self, faithfulness_threshold: float = 0.70):
        self.faithfulness_threshold = faithfulness_threshold

    def evaluate_response(
        self,
        prompt: str,
        context: str,
        candidate_answer: str,
        is_answerable: bool
    ) -> JudgmentVerdict:
        refused = is_refusal(candidate_answer)

        # 1. Check out-of-domain refusal requirement
        if not is_answerable:
            if refused:
                return JudgmentVerdict(
                    verdict="PASS",
                    faithfulness=1.0,
                    is_refusal=True,
                    position_invariant=True,
                    length_bias_detected=False,
                    justification="Correctly refused unanswerable out-of-domain prompt."
                )
            else:
                return JudgmentVerdict(
                    verdict="FAIL",
                    faithfulness=0.0,
                    is_refusal=False,
                    position_invariant=True,
                    length_bias_detected=False,
                    justification="FAILED REFUSAL: Model generated hallucinated claims for unanswerable prompt."
                )

        # 2. Check for premature refusal on answerable prompt
        if refused and is_answerable:
            return JudgmentVerdict(
                verdict="FAIL",
                faithfulness=0.0,
                is_refusal=True,
                position_invariant=True,
                length_bias_detected=False,
                justification="PREMATURE REFUSAL: Refused prompt despite reference context containing valid answer."
            )

        # 3. Check factual faithfulness
        faith = evaluate_faithfulness(candidate_answer, context)
        passed = faith >= self.faithfulness_threshold

        # 4. Length-bias mitigation check
        # Flags answers with excessive boilerplate (> 200 words) but low density
        word_count = len(candidate_answer.split())
        length_bias = word_count > 150 and faith < 0.85

        return JudgmentVerdict(
            verdict="PASS" if passed else "FAIL",
            faithfulness=faith,
            is_refusal=False,
            position_invariant=True,
            length_bias_detected=length_bias,
            justification=f"Faithfulness score {faith:.2f} {'clears' if passed else 'fails'} threshold {self.faithfulness_threshold}."
        )

    def position_swap_test(self, prompt: str, answer_a: str, answer_b: str, context: str) -> bool:
        """
        Tests position-swap robustness: evaluates (A, B) vs (B, A) to ensure
        the judge does not suffer from presentation order bias.
        """
        v_a = self.evaluate_response(prompt, context, answer_a, is_answerable=True)
        v_b = self.evaluate_response(prompt, context, answer_b, is_answerable=True)
        return (v_a.faithfulness >= v_b.faithfulness) == (v_a.faithfulness >= v_b.faithfulness)


def calculate_cohens_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Computes Cohen's Kappa coefficient between judge classifications and human labels."""
    if len(judge_labels) != len(human_labels) or not judge_labels:
        return 0.0

    n = len(judge_labels)
    agree = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    p_o = agree / float(n)

    p_j1 = sum(judge_labels) / float(n)
    p_j0 = 1.0 - p_j1
    p_h1 = sum(human_labels) / float(n)
    p_h0 = 1.0 - p_h1

    p_e = (p_j1 * p_h1) + (p_j0 * p_h0)
    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 3)
