"""Observability - log-based metric-events for Datadog (stdout JSON).

Each metric is exactly 1 JSON line on stdout:

    {"log_type": "metric", "event": "<name>", "status": "info",
     "service": "<svc>", "env": "<env>", ...tags/values}

Stream (log_type:metric), supplements structured log structlog + Langfuse.
Does NOT replace. Datadog parses `@log_type:metric @event:*` -> log-based
metric ai_lesson_builder.*. `emit()` NEVER raises.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Mapping

SERVICE: str = os.getenv("DD_SERVICE", "ai-lesson-builder-service")
ENV: str = os.getenv("DD_ENV") or os.getenv("ENVIRONMENT") or "production"

_logger = logging.getLogger("metric_events")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

_SPINE_HEADERS = {
    "x-robot-id": "robot_id",
    "x-conversation-id": "conversation_id",
    "x-user-id": "user_id",
    "x-bot-id": "bot_id",
}


def emit(event: str, **fields: Any) -> None:
    """Write 1 metric event to stdout as single-line JSON. Never raises."""
    try:
        payload: dict[str, Any] = {
            "log_type": "metric",
            "event": event,
            "status": "info",
            "service": SERVICE,
            "env": ENV,
        }
        for key, value in fields.items():
            if value is None:
                continue
            if key.endswith("_ms") and isinstance(value, float):
                payload[key] = round(value, 1)
            else:
                payload[key] = value
        _logger.info(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        pass


def outcome_for(status_code: int) -> str:
    """HTTP status -> outcome enum."""
    if status_code >= 500:
        return "http_5xx"
    if status_code >= 400:
        return "http_4xx"
    return "ok"


def spine_from_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Extract spine from request headers."""
    out: dict[str, str] = {}
    try:
        for header, field in _SPINE_HEADERS.items():
            val = headers.get(header)
            if val:
                out[field] = val
    except Exception:  # noqa: BLE001
        pass
    return out


def route_template(request: Any) -> str:
    """Route template (no IDs); fallback url.path."""
    try:
        route = request.scope.get("route")
        return getattr(route, "path", None) or str(request.url.path)
    except Exception:  # noqa: BLE001
        return "unknown"
