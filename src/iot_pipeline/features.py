"""Time-series feature engineering for IoT sensor telemetry."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "value",
    "rolling_mean_6",
    "rolling_std_6",
    "z_score",
    "delta_1",
    "hour_sin",
    "hour_cos",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sensor rolling statistics, deltas, and cyclic time features.

    Expects columns: `sensor_id`, `timestamp`, `value`.
    Returns a copy sorted by (`sensor_id`, `timestamp`) with `FEATURE_COLUMNS`.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    grouped = work.groupby("sensor_id", sort=False)["value"]
    work["rolling_mean_6"] = grouped.transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    work["rolling_std_6"] = (
        grouped.transform(lambda s: s.rolling(window=6, min_periods=1).std(ddof=0))
        .fillna(0.0)
    )
    safe_std = np.where(work["rolling_std_6"] > 1e-6, work["rolling_std_6"], 1.0)
    work["z_score"] = (work["value"] - work["rolling_mean_6"]) / safe_std
    work["delta_1"] = grouped.diff().fillna(0.0)

    hours = work["timestamp"].dt.hour + work["timestamp"].dt.minute / 60.0
    work["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    work["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)

    return work
