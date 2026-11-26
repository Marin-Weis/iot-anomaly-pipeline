"""DuckDB analytical SQL storage for sensor readings and anomaly predictions."""

from pathlib import Path

import duckdb
import pandas as pd

from iot_pipeline.schemas import PredictionRecord, SensorReading, SensorSummary


class DuckDBStorage:
    """Manages time-series readings and anomaly predictions in DuckDB."""

    def __init__(self, db_path: str | Path = "data/iot_telemetry.duckdb") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    sensor_id VARCHAR NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    value DOUBLE NOT NULL,
                    battery_pct DOUBLE NOT NULL,
                    signal_rssi INTEGER NOT NULL,
                    PRIMARY KEY (sensor_id, timestamp)
                );

                CREATE TABLE IF NOT EXISTS anomaly_predictions (
                    sensor_id VARCHAR NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    value DOUBLE NOT NULL,
                    rolling_mean DOUBLE NOT NULL,
                    z_score DOUBLE NOT NULL,
                    anomaly_score DOUBLE NOT NULL,
                    is_anomaly BOOLEAN NOT NULL,
                    severity VARCHAR NOT NULL,
                    PRIMARY KEY (sensor_id, timestamp)
                );
                """
            )

    def save_readings_and_predictions(
        self,
        readings: list[SensorReading],
        predictions: list[PredictionRecord],
    ) -> None:
        if not readings:
            return

        readings_df = pd.DataFrame([r.model_dump() for r in readings])
        preds_df = pd.DataFrame([p.model_dump() for p in predictions])

        with self._connect() as conn:
            conn.register("readings_df", readings_df)
            conn.execute(
                """
                INSERT OR REPLACE INTO sensor_readings
                SELECT sensor_id, timestamp, value, battery_pct, signal_rssi
                FROM readings_df
                """
            )
            conn.register("preds_df", preds_df)
            conn.execute(
                """
                INSERT OR REPLACE INTO anomaly_predictions
                SELECT sensor_id, timestamp, value, rolling_mean, z_score,
                       anomaly_score, is_anomaly, severity
                FROM preds_df
                """
            )

    def list_anomalies(
        self,
        sensor_id: str | None = None,
        limit: int = 100,
    ) -> list[PredictionRecord]:
        query = """
            SELECT sensor_id, timestamp, value, rolling_mean, z_score,
                   anomaly_score, is_anomaly, severity
            FROM anomaly_predictions
            WHERE is_anomaly = TRUE
        """
        params: list[object] = []
        if sensor_id:
            query += " AND sensor_id = ?"
            params.append(sensor_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchdf()

        return [PredictionRecord(**row) for row in rows.to_dict(orient="records")]

    def get_sensor_summary(self, sensor_id: str) -> SensorSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    sensor_id,
                    COUNT(*)::INTEGER AS total_readings,
                    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END)::INTEGER AS anomaly_count,
                    AVG(value) AS avg_value,
                    MAX(value) AS max_value,
                    MAX(timestamp) AS last_seen
                FROM anomaly_predictions
                WHERE sensor_id = ?
                GROUP BY sensor_id
                """,
                [sensor_id],
            ).fetchone()

        if row is None:
            return None

        total = row[1]
        anomalies = row[2]
        return SensorSummary(
            sensor_id=row[0],
            total_readings=total,
            anomaly_count=anomalies,
            anomaly_rate=round(anomalies / total, 4) if total else 0.0,
            avg_value=round(float(row[3]), 2),
            max_value=round(float(row[4]), 2),
            last_seen=row[5],
        )

    def fetch_recent_readings(self, sensor_ids: list[str], tail_per_sensor: int = 10) -> pd.DataFrame:
        if not sensor_ids:
            return pd.DataFrame(columns=["sensor_id", "timestamp", "value"])
        placeholders = ", ".join(["?"] * len(sensor_ids))
        query = f"""
            SELECT sensor_id, timestamp, value
            FROM (
                SELECT sensor_id, timestamp, value,
                       ROW_NUMBER() OVER (PARTITION BY sensor_id ORDER BY timestamp DESC) AS rn
                FROM sensor_readings
                WHERE sensor_id IN ({placeholders})
            )
            WHERE rn <= ?
            ORDER BY sensor_id, timestamp ASC
        """
        with self._connect() as conn:
            return conn.execute(query, [*sensor_ids, tail_per_sensor]).fetchdf()
