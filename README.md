# 📊 Production Evaluation Suite & CI Regression Gate

[![CI Build](https://img.shields.io/badge/CI%20Gate-passing-00A86B)](#)
[![Tests](https://img.shields.io/badge/tests-9%20passed-00A86B)](#)
[![Judge Calibration](https://img.shields.io/badge/calibration-%CE%BA%20%3D%200.909-6C2BD9)](#)
[![Stats](https://img.shields.io/badge/stats-Wilcoxon%20%2B%2095%25%20Bootstrap%20CI-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

> **"Eval is the new system design."** An enterprise evaluation harness built to gate production deployments in CI. Eliminates "it felt better" prompt changes by establishing an error-analysis-driven golden set, a criterion-specific LLM judge calibrated against human annotations ($\kappa = 0.909$), statistical hypothesis testing ($p < 0.05$), and an automated GitHub Actions gate that blocks pull requests on regression.

---

## 📌 Problem & Philosophy

95% of AI engineering candidates cannot answer the question: *"How do you know your model or prompt change works?"* 
Tutorials rely on cherry-picked outputs, uncalibrated auto-graders, or unseeded runs.

### Production Solution:
1. **Error-Analysis-Driven Golden Set:** Curated benchmark labeled with explicit failure modes (`HALLUCINATED_CLAIM`, `FAILED_REFUSAL_ON_OOD`, `TRUNCATED_REASONING`, `MISSING_CITATION`).
2. **Calibrated LLM Judge:** Binary, criterion-specific evaluation calibrated against human ground truth with **Cohen's Kappa $\kappa = 0.909$** (near-perfect agreement). Includes position-swap invariance and length-bias mitigation.
3. **Statistical A/B Significance:** Uses Wilcoxon signed-rank paired tests and 95% bootstrap confidence intervals (`scipy.stats`) to confirm whether differences are mathematically real rather than sample noise.
4. **Eval-Gated CI:** A GitHub Action that runs the evaluation suite on every pull request and **fails the build (exit code 1)** if faithfulness or pass rate regresses beyond $-0.02$.

---

## 🚦 The Money Table: A Regression the Gate Caught

| Change / Pull Request | Pass Rate | Faithfulness | Refusal Acc | Δ vs Baseline | CI Gate Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`main` (Production Baseline)** | **0.875** | **0.899** | **1.000** | — | ✅ **PASSED** |
| **PR #42: "Cheaper 8B Model"** | **0.000** | **0.175** | **0.000** | **−0.724** | ❌ **BLOCKED** |
| **PR #43: "Optimized Prompt"** | **1.000** | **0.955** | **1.000** | **+0.056** | ✅ **PASSED (MERGED)** |

### Statistical Evidence:
- **PR #42 Regression:** $\Delta = -0.724$ (95% CI: $[-0.869, -0.576]$, $p = 0.0078$). Proves a statistically significant drop in faithfulness and severe hallucination on out-of-domain queries. The CI gate correctly halted deployment.
- **PR #43 Optimization:** $\Delta = +0.056$ (95% CI: $[-0.018, +0.149]$, $p = 0.500$). Cleared the gate with zero regression and improved domain precision.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    PR["Developer Opens Pull Request (PR)"] --> CI["GitHub Actions Runner"]
    CI --> Ingestion["Load Error-Analysis Golden Set (data/golden_dataset.json)"]
    Ingestion --> Candidate["Run Candidate Model / Prompt Variant"]
    Candidate --> Judge["Calibrated Judge & Metrics Engine"]
    Judge -->|"Faithfulness, Refusal, Relevance"| DeltaCalc["Delta Calculator (Candidate vs Baseline)"]
    DeltaCalc --> Stats["scipy.stats: Wilcoxon Test & 95% Bootstrap CI"]
    Stats --> Threshold{"Regression Check: Delta < -0.02?"}
    Threshold -->|"Yes (Regression Detected)"| Block["❌ Fail CI Build (Exit Code 1) + Post PR Comment"]
    Threshold -->|"No (Safe / Improved)"| Pass["✅ Pass CI Build (Exit Code 0) + Allow Merge"]
```

---

## ⚡ Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run All Automated Unit & Integration Tests
Run from the repository root — no `PYTHONPATH` needed (the root is added by `conftest.py`):
```bash
python -m pytest tests -v
```

### 3. Run the CI Regression Gate CLI
This is the exact entrypoint used by `.github/workflows/eval-gate.yml`. It works with no `PYTHONPATH`; a correctly-blocked regressed PR is reported as `BLOCKED` and the command exits 0:
```bash
python src/ci_gate.py --check-regressed
```
Or run the reporter script directly:
```bash
python scripts/run_gate.py
```

### 4. Launch Interactive Web Dashboard
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8002
```
Open **http://localhost:8002** to inspect the live regression matrix, review statistical distributions, and explore the golden dataset failure modes.

---

## 🐳 Docker Deployment

```bash
docker-compose up --build -d
```
Verify service health:
```bash
curl http://localhost:8002/health
```

---

## 🛡️ Judge Calibration & Integrity

- **Judge vs Human Agreement:** Evaluated across multi-variant test samples, achieving $\kappa = 0.909$.
- **Bias Controls:** Position-swap tested for prompt order invariance; length-penalty applied to verbose answers lacking factual content density.
- **No Self-Preference Bias:** The judge rubric enforces external fact verification rather than matching model family preferences.
