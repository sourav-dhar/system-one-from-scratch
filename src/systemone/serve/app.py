"""A ``POST /v1/systemone`` server for any decision model of this harness (TypeSafe-compatible).

    from systemone.serve.app import create_app
    app = create_app(model, name="systemone-m1")      # then: uvicorn module:app

Routes:
* ``GET /health``: ``{"status": "ok", "model": name}``.
* ``GET /v1/models``: the served model, in the usual ``{"object": "list", "data": [...]}`` shape.
* ``POST /v1/systemone``: the wire format in :mod:`systemone.serve.wire`.

Malformed bodies get FastAPI's 422 ``detail`` list, which is the documented
``HTTPValidationError`` shape. The server answers 400 ``{"detail": ...}`` to:
* an unknown model name;
* image assets;
* an input the model refuses, such as one longer than its context.

With ``api_key`` set, requests need ``Authorization: Bearer <api_key>``; otherwise the server answers
401.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from systemone.bench.runner import DecisionModel
from systemone.serve.wire import (
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    to_decision_request,
    to_wire_answer,
)


def create_app(model: DecisionModel, name: str | None = None, api_key: str | None = None) -> Any:
    """A FastAPI app serving ``model`` under ``name`` (default: its model id). Needs the ``serve`` extra."""
    from fastapi import FastAPI, Header, HTTPException

    served = name or model.model_id
    app = FastAPI(title="System One", summary=f"Typed decisions from {served}", version="0.1.0")

    def check(authorization: str | None) -> None:
        if api_key is None:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(token, api_key):
            raise HTTPException(401, "invalid or missing API key")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": served}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        return {"object": "list", "data": [{"id": served, "object": "model", "owned_by": "systemone"}]}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(
        body: SystemOneRequest, authorization: str | None = Header(default=None)
    ) -> SystemOneResponse:
        check(authorization)
        if body.model != served:
            raise HTTPException(400, f"unknown model {body.model!r}; available: [{served!r}]")
        if body.assets:
            raise HTTPException(400, "image assets are not supported by this server")
        request, fixed = to_decision_request(body)
        probs: dict[str, dict[str, float]] = dict(fixed)
        input_tokens = 0
        if request is not None:
            started = time.perf_counter()
            (response,) = model.decide_many([request])
            missing = [n for n in request.questions if n not in response.answers]
            if missing:
                raise HTTPException(400, response.error or f"no answer for {missing}")
            probs |= {n: a.probs for n, a in response.answers.items()}
            input_tokens = int((response.usage or {}).get("input_tokens", 0))
            app.state.last_latency_ms = (time.perf_counter() - started) * 1000
        answers = {name: to_wire_answer(q, probs[name]) for name, q in body.questions.items()}
        return SystemOneResponse(model=served, answers=answers, usage=Usage(input_tokens=input_tokens))

    return app
