from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_monitoring_log_path


class PredictionMonitor:
    """Persist prediction outcomes and operational statistics, never input features."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or get_monitoring_log_path()
        self._lock = threading.Lock()

    def record_prediction(
        self,
        prediction: int,
        poisonous_probability: float,
        latency_ms: float,
        request_id: str | None = None,
    ) -> str:
        if prediction not in (0, 1):
            raise ValueError("prediction must be 0 (edible) or 1 (poisonous).")
        if not math.isfinite(poisonous_probability) or not 0 <= poisonous_probability <= 1:
            raise ValueError("poisonous_probability must be finite and between 0 and 1.")
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative.")

        identifier = request_id or uuid.uuid4().hex[:12]
        self._append(
            {
                "request_id": identifier,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "prediction": int(prediction),
                "poisonous_probability": float(poisonous_probability),
                "latency_ms": float(latency_ms),
            }
        )
        return identifier

    def record_failure(
        self, error_code: str, request_id: str | None = None
    ) -> str:
        if not error_code or not error_code.isascii():
            raise ValueError("error_code must be a non-empty ASCII identifier.")
        identifier = request_id or uuid.uuid4().hex[:12]
        self._append(
            {
                "request_id": identifier,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failure",
                "error_code": error_code,
            }
        )
        return identifier

    def summary(self) -> dict[str, Any]:
        events = self._read_events()
        predictions = [event for event in events if event.get("status") == "success"]
        failures = sum(event.get("status") == "failure" for event in events)
        poisonous_probabilities = [
            float(event["poisonous_probability"]) for event in predictions
        ]
        latencies = [float(event["latency_ms"]) for event in predictions]
        prediction_counts = {
            "edible": sum(event["prediction"] == 0 for event in predictions),
            "poisonous": sum(event["prediction"] == 1 for event in predictions),
        }
        return {
            "total_predictions": len(predictions),
            "failed_requests": failures,
            "prediction_counts": prediction_counts,
            "mean_poisonous_probability": (
                sum(poisonous_probabilities) / len(poisonous_probabilities)
                if poisonous_probabilities
                else None
            ),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "max_latency_ms": max(latencies) if latencies else None,
            "accuracy": None,
            "accuracy_note": "Accuracy is not computed without ground-truth labels.",
        }

    def _append(self, event: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(serialized + "\n")

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        events = []
        with self._lock:
            with self.log_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid monitoring record on line {line_number}."
                        ) from error
                    if not isinstance(event, dict):
                        raise ValueError(
                            f"Invalid monitoring record on line {line_number}."
                        )
                    events.append(event)
        return events
