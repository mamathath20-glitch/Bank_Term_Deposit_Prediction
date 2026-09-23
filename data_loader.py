"""
data_loader.py — loads and lightly prepares the bank marketing dataset.

Key decisions (all documented here for transparency):

1. delimiter is ';' (semicolon) — NOT comma.
2. 'duration' is DROPPED before the DataFrame is returned because it is
   only known after a call ends and would cause data leakage in any
   predictive model.
3. 'pdays' original values are PRESERVED exactly (-1 means never contacted).
   A separate binary feature 'was_contacted' is added: 1 if the customer
   was previously contacted, 0 otherwise.
4. 'y' (target) is mapped to integers: {'no': 0, 'yes': 1}.
5. 'unknown' string values are kept as valid category levels. They are
   audited and printed so the analyst can make an informed decision about
   how to handle them downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly (python src/data_loader.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PATH  # noqa: E402


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the bank marketing dataset and apply minimal, documented transforms.

    Parameters
    ----------
    path : Path | str | None
        Override the default DATA_PATH from config. Useful for testing.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for EDA and preprocessing.
        - 'duration' column is dropped (data leakage).
        - 'was_contacted' column added (1 if previously contacted, 0 otherwise).
        - 'pdays' values preserved exactly as loaded (including -1).
        - 'y' is int (0 = no, 1 = yes).
    """
    csv_path = Path(path) if path is not None else DATA_PATH

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, sep=";")
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from '{csv_path.name}'")

    # ── Drop data-leakage column ──────────────────────────────────────────────
    df = df.drop(columns=["duration"])
    print("Dropped 'duration' column (only known after call ends — data leakage).")

    # ── Encode target ─────────────────────────────────────────────────────────
    df["y"] = df["y"].map({"no": 0, "yes": 1})
    print(f"Target 'y' encoded: {df['y'].value_counts().to_dict()}")

    # ── Add was_contacted feature (pdays preserved) ───────────────────────────
    df["was_contacted"] = (df["pdays"] != -1).astype(int)
    n_contacted = df["was_contacted"].sum()
    print(
        f"Added 'was_contacted': {n_contacted:,} previously contacted "
        f"({n_contacted / len(df) * 100:.1f}%), "
        f"{len(df) - n_contacted:,} never contacted (pdays == -1)."
    )
    print("Note: 'pdays' original values preserved intact (including -1 sentinel).")

    # ── Audit 'unknown' string values ─────────────────────────────────────────
    audit = _audit_unknown(df)
    if audit.empty:
        print("\nNo 'unknown' string values found in any column.")
    else:
        print("\n--- 'unknown' value counts by column ---")
        print(audit.to_string(index=False))
        print(
            "Note: 'unknown' values are retained as valid category levels. "
            "They may carry predictive signal and are NOT treated as NaN."
        )

    return df


def _audit_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame listing columns that contain the string 'unknown'."""
    records = []
    for col in df.select_dtypes(include="object").columns:
        n = (df[col] == "unknown").sum()
        if n > 0:
            pct = n / len(df) * 100
            records.append(
                {
                    "column": col,
                    "unknown_count": n,
                    "unknown_pct": f"{pct:.1f}%",
                }
            )
    return pd.DataFrame(records, columns=["column", "unknown_count", "unknown_pct"])


def get_unknown_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper — returns the 'unknown' audit table for a given DataFrame."""
    return _audit_unknown(df)


if __name__ == "__main__":
    df = load_data()
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))
