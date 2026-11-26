"""Scikit-learn Isolation Forest anomaly detector with joblib serialization."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iot_pipeline.features import FEATURE_COLUMNS, build_features


class AnomalyDetector:
    """Wraps a StandardScaler + IsolationForest pipeline for IoT telemetry."""

    def __init__(self, contamination: float = 0.03, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=150,
                        contamination=contamination,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        featured = build_features(df)
        self.pipeline.fit(featured[FEATURE_COLUMNS])
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before calling predict()")

        featured = build_features(df)
        X = featured[FEATURE_COLUMNS]
        raw_preds = self.pipeline.predict(X)  # -1 = anomaly, 1 = normal
        # Invert decision_function so higher score = more anomalous
        scores = -self.pipeline.decision_function(X)

        featured["anomaly_score"] = np.round(scores, 4)
        featured["is_anomaly"] = raw_preds == -1
        featured["severity"] = [
            _classify_severity(is_anom, float(score))
            for is_anom, score in zip(featured["is_anomaly"], featured["anomaly_score"])
        ]
        return featured

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        return joblib.load(Path(path))


def _classify_severity(is_anomaly: bool, score: float) -> str:
    if not is_anomaly:
        return "NORMAL"
    return "CRITICAL" if score >= 0.12 else "WARNING"
