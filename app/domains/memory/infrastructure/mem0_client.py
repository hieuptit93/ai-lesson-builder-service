import time

import httpx
import structlog

logger = structlog.get_logger()


def _emit_mem0(outcome: str, start: float, *, status_code: int | None = None, error_type: str | None = None) -> None:
    """Metric-event: mem0 search_facts call (latency/outcome). Never raises."""
    try:
        from app.core.metric_events import emit as _metric_emit
        _metric_emit(
            "mem0_request",
            target_service="mem0",
            endpoint="/search_facts",
            outcome=outcome,
            status_code=status_code,
            error_type=error_type,
            latency_ms=float((time.monotonic() - start) * 1000.0),
        )
    except Exception:  # noqa: BLE001
        pass


class Mem0Client:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def search_facts(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        _start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/search_facts",
                    json={
                        "query": query,
                        "user_id": user_id,
                        "top_k": top_k,
                        "limit": limit,
                        "score_threshold": score_threshold,
                    },
                    headers={"Content-Type": "application/json", "accept": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            _emit_mem0("timeout", _start, error_type=type(exc).__name__)
            raise
        except Exception as exc:  # noqa: BLE001
            _emit_mem0("error", _start, error_type=type(exc).__name__)
            raise
        _emit_mem0("ok", _start, status_code=response.status_code)
        return data.get("facts", data) if isinstance(data, dict) else data
