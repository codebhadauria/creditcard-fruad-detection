"""Leakage-aware unsupervised anomaly detection and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_detector(model_name: str, contamination: float, random_state: int = 42) -> Pipeline:
    if not 0.0001 <= contamination <= 0.5:
        raise ValueError("contamination must be between 0.0001 and 0.5")
    if model_name == "isolation_forest":
        detector: Any = IsolationForest(
            n_estimators=250, contamination=contamination, random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "lof":
        detector = LocalOutlierFactor(n_neighbors=25, contamination=contamination, novelty=True, n_jobs=-1)
    else:
        raise ValueError("model_name must be 'isolation_forest' or 'lof'")
    return Pipeline([("scaler", StandardScaler()), ("detector", detector)])


def fit_score(frame: pd.DataFrame, model_name: str = "isolation_forest",
              contamination: float = 0.0017, random_state: int = 42) -> dict[str, Any]:
    """Fit on training normals, score holdout; higher anomaly_score means stranger."""
    if frame.empty:
        raise ValueError("No rows to model.")
    has_labels = "Class" in frame.columns
    y = frame["Class"].astype(int) if has_labels else None
    X = frame.drop(columns=["Class"], errors="ignore").select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric features available to train on.")

    indices = np.arange(len(frame))
    if has_labels and y.nunique() > 1 and y.value_counts().min() >= 2:
        train_idx, test_idx = train_test_split(
            indices, test_size=0.25, random_state=random_state, stratify=y
        )
        normal_train_idx = train_idx[y.iloc[train_idx].to_numpy() == 0]
        if len(normal_train_idx) < 2:
            raise ValueError("At least two legitimate transactions are needed for training.")
        fit_X = X.iloc[normal_train_idx]
    else:
        # With no usable labels, retain a holdout for scoring but train on all input rows.
        test_size = max(1, int(round(len(frame) * 0.25))) if len(frame) > 4 else 1
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        fit_X = X.iloc[train_idx]

    pipeline = build_detector(model_name, contamination, random_state)
    pipeline.fit(fit_X)
    # sklearn outlier detectors use lower decision_function values for anomalies.
    train_scores = -pipeline.decision_function(fit_X)
    threshold = float(np.quantile(train_scores, 1 - contamination))
    test_scores = -pipeline.decision_function(X.iloc[test_idx])
    predictions = (test_scores >= threshold).astype(int)

    scored = frame.iloc[test_idx].copy().reset_index(drop=True)
    scored["anomaly_score"] = test_scores
    scored["is_anomaly"] = predictions
    metrics: dict[str, Any] = {
        "model": model_name,
        "contamination": contamination,
        "threshold": threshold,
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_idx)),
        "rows_scored": int(len(test_idx)),
        "anomalies_flagged": int(predictions.sum()),
        "features": list(X.columns),
    }
    if has_labels:
        y_test = y.iloc[test_idx].to_numpy()
        if len(np.unique(y_test)) > 1:
            metrics.update({
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "average_precision": float(average_precision_score(y_test, test_scores)),
                "roc_auc": float(roc_auc_score(y_test, test_scores)),
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
            })
        else:
            metrics["evaluation_note"] = "The held-out split contains one class; class metrics are undefined."
    return {"pipeline": pipeline, "threshold": threshold, "metrics": metrics, "scored": scored}
