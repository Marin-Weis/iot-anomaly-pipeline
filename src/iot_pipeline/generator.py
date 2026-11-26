"""Realistic synthetic IoT flow-counter telemetry generator."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

DEFAULT_SENSORS = ["counter-north-01", "counter-port-02", "counter-campus-03"]


def generate_telemetry(
    days: int = 14,
    freq_minutes: int = 15,
    sensors: list[str] | None = None,
    inject_anomalies: bool = False,
    seed: int = 42,
    start_time: datetime | None = None,
) -> pd.DataFrame:
    """Generate realistic diurnal flow-counter telemetry (e.g. pedestrian/bike counters)."""
    rng = np.random.default_rng(seed)
    sensor_list = sensors or DEFAULT_SENSORS
    start = start_time or (datetime(2026, 11, 1, 0, 0, tzinfo=UTC))
    steps = int(days * 24 * (60 / freq_minutes))
    timestamps = [start + timedelta(minutes=freq_minutes * i) for i in range(steps)]

    rows = []
    for idx, sensor_id in enumerate(sensor_list):
        base_scale = 45.0 + idx * 18.0
        for step_i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60.0
            # Two commuting peaks (morning + late afternoon) + quiet night
            diurnal = (
                np.exp(-((hour - 8.5) ** 2) / 6.0) * 0.85
                + np.exp(-((hour - 17.5) ** 2) / 8.0) * 1.0
            )
            weekend_factor = 0.72 if ts.weekday() >= 5 else 1.0
            noise = rng.normal(0.0, 4.0)
            value = max(0.0, base_scale * (0.12 + diurnal) * weekend_factor + noise)
            rows.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": ts,
                    "value": round(float(value), 2),
                    "battery_pct": round(float(max(20.0, 98.0 - step_i * 0.01)), 1),
                    "signal_rssi": int(rng.integers(-82, -62)),
                }
            )

    df = pd.DataFrame(rows)
    if inject_anomalies and not df.empty:
        # Inject sharp sensor spikes and sudden zero-dropouts during peak hours
        for sensor_id in sensor_list:
            mask = df["sensor_id"] == sensor_id
            indices = df.index[mask].tolist()
            if len(indices) >= 40:
                df.loc[indices[len(indices) // 3], "value"] = round(base_scale * 4.8, 2)
                df.loc[indices[len(indices) // 3 + 1], "value"] = round(base_scale * 4.2, 2)
                df.loc[indices[(2 * len(indices)) // 3], "value"] = 0.0
                df.loc[indices[min(len(indices) - 5, (4 * len(indices)) // 5)], "value"] = round(
                    base_scale * 5.4, 2
                )
    return df
