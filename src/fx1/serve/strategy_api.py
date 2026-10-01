"""Bounded strategy replay HTTP pilot; generation and semantic grading are absent.

This product API stays separate from the harness API to preserve the one-way
architecture boundary. It accepts caller-provided generated artifacts, executes
only the bounded AST interpreter, returns a reproducible receipt, and never
reads client-supplied filesystem paths, starts training, or calls a provider.
"""

from __future__ import annotations

import hmac
import os
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from fx1 import __version__
from fx1.strategy import GeneratedCode, ReplayBar, StrategySpec, backtest, build_receipt, evaluate

_MAX_REQUEST_BYTES = 64 * 1024


class _TooLarge(Exception):
    pass


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SpecInput(_Input):
    id: StrictStr = Field(min_length=1, max_length=128)
    title: StrictStr = Field(min_length=1, max_length=256)
    description: StrictStr = Field(min_length=1, max_length=16_384)
    parameters: dict[str, StrictFloat | StrictInt | StrictStr] = Field(default_factory=dict)


class GeneratedInput(_Input):
    spec_id: StrictStr = Field(min_length=1, max_length=128)
    spec_sha256: StrictStr = Field(pattern="^[0-9a-f]{64}$")
    code_text: StrictStr = Field(max_length=16_384)
    code_sha256: StrictStr = Field(pattern="^[0-9a-f]{64}$")
    generated_at: StrictStr = Field(max_length=64)
    backend_id: StrictStr = Field(min_length=1, max_length=256)
    generation_seconds: StrictFloat = Field(ge=0)
    contract: StrictStr = "fx1_strategy_expression_v1"


class BarInput(_Input):
    event_time: datetime
    available_time: datetime
    close: StrictFloat = Field(gt=0)


class ReplayInput(_Input):
    spec: SpecInput
    generated: GeneratedInput
    bars: list[BarInput] = Field(min_length=2, max_length=512)
    decision_times: list[datetime] = Field(min_length=1, max_length=128)
    data_source: StrictStr = Field(min_length=1, max_length=256)
    synthetic: StrictBool
    cost_bps: StrictFloat = Field(default=0.0, ge=0)


def create_app() -> FastAPI:
    app = FastAPI(
        title="fx-1 bounded strategy research pilot",
        version=__version__,
        description="Caller-provided code replay only; no provider, trained model, or live orders.",
    )

    @app.middleware("http")
    async def access_and_size(request: Request, call_next: Any) -> Any:
        if request.url.path != "/health":
            expected = os.environ.get("FX1_API_KEY")
            provided = request.headers.get("X-API-Key", "")
            host = request.client.host if request.client else ""
            if expected and not hmac.compare_digest(provided.encode(), expected.encode()):
                return JSONResponse(status_code=401, content={"detail": "invalid X-API-Key"})
            if not expected and host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
                return JSONResponse(status_code=403, content={"detail": "loopback only"})
        received = 0
        original_receive = request._receive

        async def receive() -> Any:
            nonlocal received
            message = await original_receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > _MAX_REQUEST_BYTES:
                    raise _TooLarge
            return message

        request._receive = receive
        try:
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    count = int(length)
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "invalid length"})
                if count < 0:
                    return JSONResponse(status_code=400, content={"detail": "invalid length"})
                if count > _MAX_REQUEST_BYTES:
                    raise _TooLarge
            response = await call_next(request)
            # FastAPI can translate a receive exception to its generic 400.
            # Preserve the boundary's precise size error after that translation.
            if received > _MAX_REQUEST_BYTES:
                raise _TooLarge
        except _TooLarge:
            response = JSONResponse(status_code=413, content={"detail": "64 KiB input limit"})
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "research_only": True, "live_pnl_claim": False}

    @app.get("/v1/strategy/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "schema_version": "fx1_strategy_api_v1",
            "contract": "fx1_strategy_expression_v1",
            "replay": True,
            "generation": False,
            "semantic_judge": False,
            "trained_checkpoint": False,
            "max_body_bytes": _MAX_REQUEST_BYTES,
            "max_bars": 512,
            "max_decisions": 128,
            "research_only": True,
            "live_pnl_claim": False,
        }

    @app.post("/v1/strategy/replay")
    def replay(payload: ReplayInput) -> dict[str, Any]:
        try:
            spec = StrategySpec(**payload.spec.model_dump())
            generated = GeneratedCode(**payload.generated.model_dump())
            result = backtest(
                generated,
                [ReplayBar(**row.model_dump()) for row in payload.bars],
                decision_times=payload.decision_times,
                data_source=payload.data_source,
                synthetic=payload.synthetic,
                cost_bps=payload.cost_bps,
            )
            assessment = evaluate(spec, generated, result)
            return build_receipt(spec, generated, result, assessment)
        except (ValueError, TypeError, ArithmeticError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
