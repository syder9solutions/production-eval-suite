"""
FastAPI Server for Production Eval Suite & CI Regression Gate.
Exposes endpoints for running CI evaluation gates, statistical A/B tests,
and inspecting the error-analysis golden dataset.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ci_gate import CIEvaluationGate

app = FastAPI(
    title="Production Eval Suite & CI Gate",
    description="Evaluation platform with calibrated LLM judge, statistical A/B significance, and CI regression blocking.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gate = CIEvaluationGate()
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = WEB_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>Eval Suite UI loading...</h1>")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/api/dataset")
def get_dataset():
    return {
        "total_test_cases": len(gate.dataset),
        "test_cases": gate.dataset
    }


@app.post("/api/eval/gate")
def run_ci_gate():
    report = gate.run_full_suite()
    return report


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "dataset_size": len(gate.dataset),
        "calibrated_threshold": gate.judge.faithfulness_threshold
    }
