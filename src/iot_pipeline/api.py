"""FastAPI inference and ingestion service for IoT anomaly detection."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from iot_pipeline import __version__
from iot_pipeline.generator import generate_telemetry
from iot_pipeline.model import AnomalyDetector
from iot_pipeline.schemas import (
    IngestBatchRequest,
    IngestBatchResponse,
    PredictionRecord,
    SensorSummary,
    TrainRequest,
    TrainResponse,
)
from iot_pipeline.storage import DuckDBStorage

DEFAULT_DB_PATH = os.getenv("IOT_DB_PATH", "data/iot_telemetry.duckdb")
DEFAULT_MODEL_PATH = os.getenv("IOT_MODEL_PATH", "models/isolation_forest.joblib")


def create_app(
    db_path: str | Path = DEFAULT_DB_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> FastAPI:
    storage = DuckDBStorage(db_path)
    model_file = Path(model_path)

    def _get_or_train_detector() -> AnomalyDetector:
        if model_file.exists():
            return AnomalyDetector.load(model_file)
        baseline = generate_telemetry(days=14, inject_anomalies=False, seed=42)
        det = AnomalyDetector(contamination=0.03, random_state=42).fit(baseline)
        det.save(model_file)
        return det

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = _get_or_train_detector()
        app.state.storage = storage
        yield

    app = FastAPI(
        title="IoT Anomaly Detection Pipeline API",
        version=__version__,
        description="Real-time IoT time-series ingestion, strict Pydantic validation, Isolation Forest scoring, and DuckDB SQL storage.",
        lifespan=lifespan,
    )
    # Initialize state eagerly so TestClient works with or without lifespan context
    app.state.detector = _get_or_train_detector()
    app.state.storage = storage

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "model_fitted": str(app.state.detector.is_fitted)}

    @app.post("/api/v1/train", response_model=TrainResponse)
    def train(request: TrainRequest) -> TrainResponse:
        baseline = generate_telemetry(days=request.days, inject_anomalies=False, seed=request.seed)
        detector = AnomalyDetector(contamination=request.contamination, random_state=request.seed).fit(baseline)
        detector.save(model_file)
        app.state.detector = detector
        return TrainResponse(
            trained_samples=len(baseline),
            contamination=request.contamination,
            model_path=str(model_file),
        )

    @app.post("/api/v1/ingest", response_model=IngestBatchResponse)
    def ingest(request: IngestBatchRequest) -> IngestBatchResponse:
        incoming_df = pd.DataFrame([r.model_dump() for r in request.readings])
        sensor_ids = incoming_df["sensor_id"].unique().tolist()
        history_df = app.state.storage.fetch_recent_readings(sensor_ids, tail_per_sensor=5)

        if not history_df.empty:
            history_df["is_history"] = True
            incoming_df["is_history"] = False
            combined = pd.concat(
                [history_df[["sensor_id", "timestamp", "value", "is_history"]], incoming_df],
                ignore_index=True,
            )
            scored_all = app.state.detector.predict(combined)
            scored = scored_all[~scored_all["is_history"]].copy()
        else:
            scored = app.state.detector.predict(incoming_df)

        predictions = [
            PredictionRecord(
                sensor_id=str(row["sensor_id"]),
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                value=float(row["value"]),
                rolling_mean=round(float(row["rolling_mean_6"]), 3),
                z_score=round(float(row["z_score"]), 3),
                anomaly_score=float(row["anomaly_score"]),
                is_anomaly=bool(row["is_anomaly"]),
                severity=str(row["severity"]),  # type: ignore[arg-type]
            )
            for row in scored.to_dict(orient="records")
        ]

        app.state.storage.save_readings_and_predictions(request.readings, predictions)
        anomaly_count = sum(1 for p in predictions if p.is_anomaly)
        return IngestBatchResponse(
            ingested_count=len(predictions),
            anomaly_count=anomaly_count,
            predictions=predictions,
        )

    @app.get("/api/v1/anomalies", response_model=list[PredictionRecord])
    def list_anomalies(
        sensor_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[PredictionRecord]:
        return app.state.storage.list_anomalies(sensor_id=sensor_id, limit=limit)

    @app.get("/api/v1/sensors/{sensor_id}/summary", response_model=SensorSummary)
    def sensor_summary(sensor_id: str) -> SensorSummary:
        summary = app.state.storage.get_sensor_summary(sensor_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No telemetry found for sensor '{sensor_id}'")
        return summary

    return app


app = create_app()
