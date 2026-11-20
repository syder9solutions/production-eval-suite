"""
Unit tests for A/B Statistical Significance Engine.
"""
from src.ab_significance import compute_ab_significance


def test_ab_significance_with_real_regression():
    # Baseline scores (high) vs Regressed scores (low)
    base = [0.95, 0.90, 0.88, 0.92, 0.85, 0.91, 0.89, 0.94]
    regr = [0.20, 0.15, 0.30, 0.25, 0.10, 0.18, 0.22, 0.14]

    res = compute_ab_significance(base, regr, alpha=0.05)
    assert res.delta < -0.50
    assert res.is_statistically_significant
    assert res.p_value < 0.05
    assert res.ci_upper < 0  # Upper confidence interval is strictly negative


def test_ab_significance_with_identical_variants():
    base = [0.85, 0.85, 0.85, 0.85]
    res = compute_ab_significance(base, base)
    assert res.delta == 0.0
    assert not res.is_statistically_significant
    assert res.p_value == 1.0
