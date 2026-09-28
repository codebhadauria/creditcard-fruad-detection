"""Input validation and reproducible demo data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean numeric transaction frame or raise a useful error."""
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows.")
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Column names must be unique.")
    if "Class" in frame.columns:
        labels = pd.to_numeric(frame["Class"], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError("The optional Class column must contain only 0 (legitimate) and 1 (fraud).")
        frame["Class"] = labels.astype(int)
    features = frame.drop(columns=["Class"], errors="ignore")
    features = features.select_dtypes(include=[np.number])
    if features.shape[1] == 0:
        raise ValueError("No numeric transaction features were found.")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Numeric features contain missing or infinite values. Clean the CSV and try again.")
    if features.shape[1] != len(frame.drop(columns=["Class"], errors="ignore").columns):
        ignored = sorted(set(frame.drop(columns=["Class"], errors="ignore").columns) - set(features.columns))
        raise ValueError("Non-numeric feature columns found: " + ", ".join(ignored))
    return frame


def make_demo_data(n_rows: int = 2500, fraud_rate: float = 0.025, random_state: int = 42) -> pd.DataFrame:
    """Make a small, clearly synthetic transaction dataset with rare anomalies."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100.")
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_rows * fraud_rate))
    labels = np.zeros(n_rows, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)
    frame = pd.DataFrame({
        "Time": rng.uniform(0, 172800, n_rows),
        "Amount": rng.lognormal(mean=3.2, sigma=1.0, size=n_rows),
        "V1": rng.normal(0, 1, n_rows),
        "V2": rng.normal(0, 1, n_rows),
        "V3": rng.normal(0, 1, n_rows),
        "V4": rng.normal(0, 1, n_rows),
        "V5": rng.normal(0, 1, n_rows),
        "V6": rng.normal(0, 1, n_rows),
    })
    fraud = labels == 1
    # Shift several features so the synthetic anomalies are discoverable.
    frame.loc[fraud, "Amount"] *= rng.uniform(3, 10, fraud.sum())
    frame.loc[fraud, "V1"] += rng.normal(4.5, 1, fraud.sum())
    frame.loc[fraud, "V3"] -= rng.normal(4, 1, fraud.sum())
    frame["Class"] = labels
    return frame
