"""
Statistical A/B Prompt & Model Significance Engine.
Calculates:
- Paired differences between Variant A (Baseline) and Variant B (Candidate)
- 95% Bootstrap Confidence Intervals for Delta
- Wilcoxon Signed-Rank / Paired Student's t-test p-value
- Statistical Significance verdict (reject null hypothesis at alpha = 0.05)
"""
import numpy as np
from scipy import stats
from pydantic import BaseModel


class ABSignificanceResult(BaseModel):
    mean_a: float
    mean_b: float
    delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_statistically_significant: bool
    summary: str


def compute_ab_significance(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
    n_bootstrap: int = 1000
) -> ABSignificanceResult:
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("scores_a and scores_b must be non-empty and of equal length.")

    arr_a = np.array(scores_a, dtype=np.float64)
    arr_b = np.array(scores_b, dtype=np.float64)
    diffs = arr_b - arr_a

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    delta = float(np.mean(diffs))

    # Bootstrap 95% Confidence Interval for the difference in means
    rng = np.random.default_rng(seed=42)
    boot_deltas = []
    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, len(diffs), size=len(diffs))
        boot_deltas.append(np.mean(diffs[sample_indices]))

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    # Paired statistical test
    # If differences are identical (e.g. all 0), p-value is 1.0
    if np.all(diffs == 0):
        p_val = 1.0
    else:
        try:
            # Wilcoxon signed-rank test for paired non-parametric differences
            _, p_val = stats.wilcoxon(diffs)
        except Exception:
            # Fallback to paired t-test
            _, p_val = stats.ttest_rel(arr_b, arr_a)

    is_significant = bool(p_val < alpha and (ci_lower > 0 or ci_upper < 0))

    if is_significant:
        direction = "statistically significant improvement" if delta > 0 else "statistically significant regression"
        summary = f"Delta = {delta:+.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], p = {p_val:.4f}). Demonstrates {direction}."
    else:
        summary = f"Delta = {delta:+.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], p = {p_val:.4f}). Difference is within random noise (not significant)."

    return ABSignificanceResult(
        mean_a=round(mean_a, 3),
        mean_b=round(mean_b, 3),
        delta=round(delta, 3),
        ci_lower=round(ci_lower, 3),
        ci_upper=round(ci_upper, 3),
        p_value=round(float(p_val), 4),
        is_statistically_significant=is_significant,
        summary=summary
    )
