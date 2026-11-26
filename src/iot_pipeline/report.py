"""Visual anomaly report generator using Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def render_anomaly_report(
    scored_df: pd.DataFrame,
    output_path: str | Path = "docs/anomaly-report.png",
    sensor_id: str | None = None,
) -> Path:
    """Render a two-panel dark-theme report: time-series + anomalies, and anomaly score trace."""
    target_sensor = sensor_id or str(scored_df["sensor_id"].iloc[0])
    sub = scored_df[scored_df["sensor_id"] == target_sensor].copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True)
    sub = sub.sort_values("timestamp")
    anomalies = sub[sub["is_anomaly"]]

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(color="#334155", linestyle="--", linewidth=0.6, alpha=0.7)

    # Top panel: telemetry, rolling baseline, and flagged anomalies
    ax1.plot(sub["timestamp"], sub["value"], color="#38bdf8", linewidth=1.6, label="Sensor reading")
    ax1.plot(
        sub["timestamp"],
        sub["rolling_mean_6"],
        color="#a78bfa",
        linewidth=1.3,
        linestyle="--",
        label="Rolling mean (6 steps)",
    )
    if not anomalies.empty:
        ax1.scatter(
            anomalies["timestamp"],
            anomalies["value"],
            color="#f43f5e",
            s=75,
            zorder=5,
            edgecolors="#ffffff",
            linewidths=0.9,
            label=f"Flagged anomalies ({len(anomalies)})",
        )
    ax1.set_title(
        f"IoT Anomaly Detection Report — {target_sensor}",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )
    ax1.set_ylabel("Flow / Telemetry Value", color="#cbd5e1")
    ax1.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155")

    # Bottom panel: Isolation Forest anomaly score
    ax2.plot(sub["timestamp"], sub["anomaly_score"], color="#fbbf24", linewidth=1.3, label="Anomaly score")
    ax2.axhline(0.0, color="#f43f5e", linestyle=":", linewidth=1.2, label="Decision boundary (0.0)")
    ax2.fill_between(
        sub["timestamp"],
        sub["anomaly_score"],
        0.0,
        where=sub["anomaly_score"] > 0.0,
        color="#f43f5e",
        alpha=0.25,
    )
    ax2.set_ylabel("Anomaly Score", color="#cbd5e1")
    ax2.set_xlabel("UTC Timestamp", color="#cbd5e1")
    ax2.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
