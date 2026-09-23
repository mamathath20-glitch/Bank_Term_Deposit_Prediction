"""
utils.py — model persistence helpers using joblib.

joblib is scikit-learn's recommended serialisation library for fitted pipelines.
Both functions resolve paths relative to MODELS_DIR from config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR  # noqa: E402


def save_model(model: Pipeline, filename: str = "final_model.pkl") -> Path:
    """Save a fitted sklearn Pipeline to models/.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    filename : str
        Output filename (default: 'final_model.pkl').

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = MODELS_DIR / filename
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


def load_model(filename: str = "final_model.pkl") -> Pipeline:
    """Load a previously saved sklearn Pipeline from models/.

    Parameters
    ----------
    filename : str
        Filename to load (default: 'final_model.pkl').

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run the notebook or train.py first to generate the model."
        )
    model = joblib.load(path)
    print(f"Model loaded: {path}")
    return model


if __name__ == "__main__":
    # Smoke test: train a tiny LR, save, reload, check it predicts
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from src.data_loader import load_data
    from src.preprocessing import build_preprocessor, get_X_y
    from src.train import build_pipeline, split_data

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
    )
    pipeline.fit(X_train, y_train)

    path = save_model(pipeline, "smoke_test_model.pkl")
    loaded = load_model("smoke_test_model.pkl")
    preds = loaded.predict(X_test.head(5))
    print(f"Smoke test predictions: {preds}")
    path.unlink()  # clean up test file
    print("Smoke test passed — save/load cycle verified.")
