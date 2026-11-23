"""Any ``POST /v1/systemone`` server as a decision model for the harness.

The same runner, metrics and probes then score every system that speaks TypeSafe's format: hosted Jev,
Kev's and Laya's servers, CLM's own server, and our models behind :mod:`systemone.serve.app`.

    client = SystemOneClient("https://api.typesafe.ai", model="jev-latest", api_key=...)
    responses = client.decide_many([item.request for item in items])

Failures follow the runner's rules. A 400, 413 or 422 reply means the server refuses the input; it is
recorded as ``rejected: HTTP <status> ...`` and is final. Any other failure (timeouts, 429, 5xx) is
recorded without that prefix, so the next run retries it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from systemone.decisions.schema import DecisionRequest, ModelResponse
from systemone.serve.wire import to_model_response, to_wire_request

REJECTED_STATUSES = (400, 413, 422)


class SystemOneClient:
    """Needs the ``serve`` extra (httpx). ``http`` may be any httpx-compatible client, e.g. a TestClient."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        workers: int = 8,
        http: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.workers = workers
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if http is None:
            import httpx

            http = httpx.Client(timeout=timeout)
        self.http = http
        self.headers = headers

    @property
    def model_id(self) -> str:
        return f"systemone/{self.model}"

    def decide(self, request: DecisionRequest) -> ModelResponse:
        started = time.perf_counter()
        try:
            reply = self.http.post(
                f"{self.base_url}/v1/systemone",
                json=to_wire_request(request, self.model),
                headers=self.headers,
            )
        except Exception as error:  # transport failure: retried on the next run
            return ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}")
        latency = (time.perf_counter() - started) * 1000
        if reply.status_code != 200:
            prefix = "rejected: " if reply.status_code in REJECTED_STATUSES else ""
            return ModelResponse(
                error=f"{prefix}HTTP {reply.status_code}: {reply.text[:300]}", latency_ms=latency
            )
        return to_model_response(reply.json(), request, latency)

    def decide_many(self, requests: Sequence[DecisionRequest]) -> list[ModelResponse]:
        if self.workers <= 1 or len(requests) <= 1:
            return [self.decide(r) for r in requests]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.decide, requests))
