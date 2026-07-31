"""Structured logging and Prometheus-style metrics.

Both are opt-in and cheap. The metrics are process-local counters, not a
time-series database: they answer "is this instance serving traffic, how
fast, and how often is the model being overruled" without adding a
dependency or a sidecar.

Counters reset when the process restarts, which is what a Prometheus
scrape expects of a counter it will rate() anyway.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from typing import Any

from app.config import settings

# Attributes LogRecord always carries. Anything else on a record was put
# there by a caller via `extra=` and belongs in the structured output.
_STANDARD_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON.

    Log aggregators can filter on fields rather than regex over a message,
    which is what makes the request id threaded through the API useful in
    production.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Anything passed as extra={...} rides along as its own field.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS:
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install the configured log format on the root handler.

    Called once at import in app.main. LOG_FORMAT=json switches to
    structured output; the default stays human-readable so local
    development is not made worse in the name of production.
    """
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    if not root.handlers:
        root.addHandler(logging.StreamHandler())

    formatter: logging.Formatter
    if settings.LOG_FORMAT == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    for handler in root.handlers:
        handler.setFormatter(formatter)


class Metrics:
    """Process-local counters and latency totals.

    Guarded by a lock because uvicorn may serve concurrent requests in
    threads; the cost is negligible next to scoring a DataFrame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        # A plain dict, not a Counter: Counter is int-valued and these
        # are cumulative seconds.
        self._duration_seconds: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._cmls_scored = 0
        self._overrides_applied = 0
        self._started = time.monotonic()

    def observe_request(self, method: str, path: str, status_code: int, seconds: float) -> None:
        with self._lock:
            self._requests[(method, path, status_code)] += 1
            self._duration_seconds[(method, path)] += seconds

    def observe_scoring(self, cmls: int, overrides: int) -> None:
        with self._lock:
            self._cmls_scored += cmls
            self._overrides_applied += overrides

    def reset(self) -> None:
        """Only for tests; a live process never resets its counters."""
        with self._lock:
            self._requests.clear()
            self._duration_seconds.clear()
            self._cmls_scored = 0
            self._overrides_applied = 0

    def render(self) -> str:
        """Serialise to the Prometheus text exposition format."""
        with self._lock:
            requests = dict(self._requests)
            durations = dict(self._duration_seconds)
            cmls_scored = self._cmls_scored
            overrides = self._overrides_applied
            uptime = time.monotonic() - self._started

        lines = [
            "# HELP cml_requests_total HTTP requests handled.",
            "# TYPE cml_requests_total counter",
        ]
        for (method, path, status_code), count in sorted(requests.items()):
            lines.append(
                f'cml_requests_total{{method="{method}",path="{path}",'
                f'status="{status_code}"}} {count}'
            )

        lines += [
            "# HELP cml_request_duration_seconds_total Cumulative request duration.",
            "# TYPE cml_request_duration_seconds_total counter",
        ]
        for (method, path), total in sorted(durations.items()):
            lines.append(
                f'cml_request_duration_seconds_total{{method="{method}",path="{path}"}} {total:.6f}'
            )

        lines += [
            "# HELP cml_scored_total CMLs scored across all requests.",
            "# TYPE cml_scored_total counter",
            f"cml_scored_total {cmls_scored}",
            "# HELP cml_sme_overrides_applied_total Scored CMLs that carried an expert override.",
            "# TYPE cml_sme_overrides_applied_total counter",
            f"cml_sme_overrides_applied_total {overrides}",
            "# HELP cml_process_uptime_seconds Seconds since this process started serving.",
            "# TYPE cml_process_uptime_seconds gauge",
            f"cml_process_uptime_seconds {uptime:.3f}",
        ]
        return "\n".join(lines) + "\n"


#: Module-level so every request handler shares one set of counters.
metrics = Metrics()
