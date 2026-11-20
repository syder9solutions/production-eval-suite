"""Pytest configuration for the Production Evaluation Suite.

Ensures the repository root is importable so the project's ``from src...``
imports resolve when pytest is invoked via the console script
(``pytest tests -v``) with no PYTHONPATH set. Mirrors the sys.path insert used
in ``scripts/run_gate.py`` and ``src/ci_gate.py``.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
