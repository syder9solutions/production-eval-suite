#!/usr/bin/env python3
"""
CLI Script to execute the Eval-Gated CI Verification Suite.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ci_gate import CIEvaluationGate


def main():
    print("Executing Production Evaluation Regression Gate...")
    gate = CIEvaluationGate()
    report = gate.run_full_suite()
    print(gate.print_markdown_table(report))


if __name__ == "__main__":
    main()
