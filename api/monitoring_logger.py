from __future__ import annotations

from pathlib import Path

from src.mushroom_classification.monitoring import PredictionMonitor


class MonitoringLogger(PredictionMonitor):
    """Compatibility-named API facade over the shared prediction monitor."""


def create_monitor(log_path: Path | None = None) -> MonitoringLogger:
    return MonitoringLogger(log_path)
