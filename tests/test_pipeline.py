"""Unit and integration tests for the IoT anomaly detection pipeline."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iot_pipeline.api import create_app
from iot_pipeline.features import FEATURE_COLUMNS, build_features
from iot_pipeline.generator import generate_telemetry
from iot_pipeline.model import AnomalyDetector
from iot_pipeline.report import render_anomaly_report


def test_feature_engineering_adds_expected_columns() -> None:
    df = generate_telemetry(days=2, sensors=["counter-test-01"], seed=1)
    featured = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in featured.columns
        assert not featured[col].isna().any()


def test_isolation_forest_flags_injected_spike() -> None:
    baseline = generate_telemetry(days=10, sensors=["counter-north-01"], inject_anomalies=False, seed=42)
    detector = AnomalyDetector(contamination=0.02, random_state=42).fit(baseline)

    test_df = generate_telemetry(days=2, sensors=["counter-north-01"], inject_anomalies=False, seed=7)
    # Inject an extreme spike
    spike_idx = len(test_df) // 2
    test_df.loc[spike_idx, "value"] = 950.0

    scored = detector.predict(test_df)
    assert bool(scored.loc[spike_idx, "is_anomaly"]) is True
    assert scored.loc[spike_idx, "severity"] in {"WARNING", "CRITICAL"}


def test_model_save_and_load_roundtrip(tmp_path: Path) -> None:
    baseline = generate_telemetry(days=5, sensors=["s1"], seed=10)
    detector = AnomalyDetector(contamination=0.03, random_state=10).fit(baseline)
    model_file = tmp_path / "model.joblib"
    detector.save(model_file)

    loaded = AnomalyDetector.load(model_file)
    assert loaded.is_fitted is True
    scored = loaded.predict(baseline.head(10))
    assert len(scored) == 10


def test_render_visual_report(tmp_path: Path) -> None:
    df = generate_telemetry(days=3, sensors=["counter-north-01"], inject_anomalies=True, seed=5)
    detector = AnomalyDetector(contamination=0.03, random_state=5).fit(df)
    scored = detector.predict(df)

    out_png = tmp_path / "report.png"
    result = render_anomaly_report(scored, output_path=out_png, sensor_id="counter-north-01")
    assert result.exists()
    assert result.stat().st_size > 10_000


def test_fastapi_validation_ingest_and_sql_queries(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "test.duckdb",
        model_path=tmp_path / "model.joblib",
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    # Strict validation rejects negative sensor value
    bad_payload = {
        "readings": [
            {"sensor_id": "counter-01", "timestamp": "2026-11-20T10:00:00Z", "value": -5.0}
        ]
    }
    assert client.post("/api/v1/ingest", json=bad_payload).status_code == 422

    # Ingest normal readings + extreme anomaly
    t0 = datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
    readings = [
        {"sensor_id": "counter-01", "timestamp": (t0 + timedelta(minutes=15 * i)).isoformat(), "value": 42.0 + (i % 3)}
        for i in range(10)
    ]
    readings.append(
        {"sensor_id": "counter-01", "timestamp": (t0 + timedelta(minutes=150)).isoformat(), "value": 1200.0}
    )

    resp = client.post("/api/v1/ingest", json={"readings": readings})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ingested_count"] == 11
    assert body["anomaly_count"] >= 1

    anomalies = client.get("/api/v1/anomalies?sensor_id=counter-01").json()
    assert len(anomalies) >= 1
    assert anomalies[0]["value"] == pytest.approx(1200.0)

    summary = client.get("/api/v1/sensors/counter-01/summary").json()
    assert summary["total_readings"] == 11
    assert summary["anomaly_count"] >= 1
    assert client.get("/api/v1/sensors/unknown-sensor/summary").status_code == 404
