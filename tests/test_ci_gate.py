"""
Unit and integration tests for CI Evaluation Gate and API endpoints.
"""
from fastapi.testclient import TestClient
from src.ci_gate import CIEvaluationGate
from src.api import app

client = TestClient(app)


def test_ci_gate_blocks_regressed_pr():
    gate = CIEvaluationGate(max_regression_tolerance=-0.02)
    report = gate.run_full_suite()

    # main baseline must pass
    assert report["baseline"]["gate_verdict"] == "PASSED"

    # regressed PR must be blocked
    assert report["regressed_pr"]["gate_verdict"] == "BLOCKED_REGRESSION"
    assert report["regressed_pr"]["delta_faithfulness_vs_baseline"] < -0.02

    # improved PR must pass
    assert report["improved_pr"]["gate_verdict"] == "PASSED"
    assert report["improved_pr"]["delta_faithfulness_vs_baseline"] >= 0.0


def test_api_endpoints():
    # Health check
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    # Dataset inspection
    d = client.get("/api/dataset")
    assert d.status_code == 200
    assert d.json()["total_test_cases"] >= 8

    # Run gate via API
    g = client.post("/api/eval/gate")
    assert g.status_code == 200
    data = g.json()
    assert data["baseline"]["pass_rate"] > 0.80
    assert data["regressed_pr"]["gate_verdict"] == "BLOCKED_REGRESSION"
