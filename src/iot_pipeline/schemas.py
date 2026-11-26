"""Strict Pydantic v2 schemas for IoT sensor ingestion and anomaly inference."""

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """Single time-series observation from an IoT sensor."""

    sensor_id: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    timestamp: datetime
    value: float = Field(..., ge=0.0, le=100_000.0, description="Measured flow or telemetry value")
    battery_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    signal_rssi: int = Field(default=-70, ge=-140, le=0)

    @field_validator("value", "battery_pct")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Value must be finite")
        return v


class IngestBatchRequest(BaseModel):
    """Batch of sensor readings sent for validation, storage, and anomaly scoring."""

    readings: list[SensorReading] = Field(..., min_length=1, max_length=10_000)


class PredictionRecord(BaseModel):
    """Scored reading returned by inference and stored in DuckDB."""

    sensor_id: str
    timestamp: datetime
    value: float
    rolling_mean: float
    z_score: float
    anomaly_score: float
    is_anomaly: bool
    severity: Literal["NORMAL", "WARNING", "CRITICAL"]


class IngestBatchResponse(BaseModel):
    """Summary of an ingested and scored batch."""

    ingested_count: int
    anomaly_count: int
    predictions: list[PredictionRecord]


class TrainRequest(BaseModel):
    """Optional parameters for model training on stored or synthetic baseline data."""

    contamination: float = Field(default=0.03, gt=0.0, lt=0.5)
    days: int = Field(default=14, ge=3, le=90)
    seed: int = Field(default=42)


class TrainResponse(BaseModel):
    """Training run summary."""

    trained_samples: int
    contamination: float
    model_path: str


class SensorSummary(BaseModel):
    """Aggregate SQL summary for a sensor."""

    sensor_id: str
    total_readings: int
    anomaly_count: int
    anomaly_rate: float
    avg_value: float
    max_value: float
    last_seen: datetime | None
