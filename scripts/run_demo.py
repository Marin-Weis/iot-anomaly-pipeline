#!/usr/bin/env python3
"""End-to-end demo: train Isolation Forest, score telemetry batch, store in DuckDB, render report."""

from datetime import UTC, datetime
from pathlib import Path

from iot_pipeline.generator import generate_telemetry
from iot_pipeline.model import AnomalyDetector
from iot_pipeline.report import render_anomaly_report
from iot_pipeline.schemas import PredictionRecord, SensorReading
from iot_pipeline.storage import DuckDBStorage


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    db_path = root / "data/iot_telemetry.duckdb"
    model_path = root / "models/isolation_forest.joblib"
    report_path = root / "docs/anomaly-report.png"

    print("1. Generating 14 days of clean baseline telemetry...")
    train_df = generate_telemetry(days=14, inject_anomalies=False, seed=42)

    print("2. Training StandardScaler + IsolationForest pipeline...")
    detector = AnomalyDetector(contamination=0.025, random_state=42).fit(train_df)
    detector.save(model_path)

    print("3. Generating 3-day test batch with injected spikes & dropouts...")
    test_df = generate_telemetry(
        days=3,
        inject_anomalies=True,
        seed=99,
        start_time=datetime(2026, 11, 15, 0, 0, tzinfo=UTC),
    )
    scored_df = detector.predict(test_df)

    print("4. Persisting readings and predictions to DuckDB...")
    storage = DuckDBStorage(db_path)
    readings = [SensorReading(**row) for row in test_df.to_dict(orient="records")]
    predictions = [
        PredictionRecord(
            sensor_id=str(row["sensor_id"]),
            timestamp=row["timestamp"].to_pydatetime(),
            value=float(row["value"]),
            rolling_mean=round(float(row["rolling_mean_6"]), 3),
            z_score=round(float(row["z_score"]), 3),
            anomaly_score=float(row["anomaly_score"]),
            is_anomaly=bool(row["is_anomaly"]),
            severity=str(row["severity"]),  # type: ignore[arg-type]
        )
        for row in scored_df.to_dict(orient="records")
    ]
    storage.save_readings_and_predictions(readings, predictions)

    print("5. Rendering visual anomaly report...")
    out = render_anomaly_report(scored_df, output_path=report_path, sensor_id="counter-north-01")

    anomalies = storage.list_anomalies(limit=10)
    print(f"Done! Stored {len(predictions)} readings, flagged {len(anomalies)} anomalies -> {out}")


if __name__ == "__main__":
    main()
