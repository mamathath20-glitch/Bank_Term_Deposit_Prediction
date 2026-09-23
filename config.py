"""
config.py — central configuration for paths and constants.

All paths are resolved relative to the project root (parent of src/),
so the project works on any machine regardless of absolute location.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT_DIR / "dataset" / "bank-full.csv"

# ── Output directories ────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

# Ensure output directories exist when config is imported
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20
