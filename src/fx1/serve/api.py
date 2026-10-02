"""The fx-1 harness HTTP surface.

``fx1.harness`` registers every lab command the model may touch; this app
exposes that registry — plus receipt verification and backend completion —
over HTTP so an fx-1 instance drives the harness through one authenticated
socket instead of shell access.

Routes (all POST bodies are ``extra="forbid"`` — no silent arguments):

- ``GET  /health`` — liveness + which backends are configured (booleans
  only; credential values never leave this process).
- ``GET  /harness/commands`` — the registry; ``?role=`` filters by
  HarnessRole. Anything not listed is unreachable — fail-closed by
  construction, same as the in-process surface.
- ``POST /harness/runs`` — execute a registered command through
  :class:`~fx1.harness.Harness`; unknown names 404, ``configs/`` escapes
  422, and the command's own timeout bounds the call.
- ``POST /harness/complete`` — chat completion through
  ``get_backend(backend)`` wrapped by ``cited_complete`` (the honesty gate
  runs server-side, in production, on every response). Backend config
  failures 503, unknown backends 404, honesty violations 502 — the model's
  output contract is enforced before bytes leave.
- ``POST /receipts/verify`` — verify an arbitrary receipt object with
  ``verify_receipt_payload``; callers never need filesystem access to the
  evidence store.

Auth posture mirrors ``quant_fund.api.research_api``: ``/health`` is the
only unauthenticated route; when ``FX1_API_KEY`` is set every other route
requires ``X-API-Key``; unset, only loopback clients are served.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fx1 import __version__
from fx1.harness import Harness, HarnessRole
from fx1.honesty import Fx1HonestyError
from fx1.serve.backends import (
    BYOK_API_KEY_ENV,
    BYOK_BASE_URL_ENV,
    BYOK_MODEL_ENV,
    get_backend,
)
from fx1.serve.chat import cited_complete
from quant_fund.research.receipt_v2 import verify_receipt_payload

_API_KEY_ENV = "FX1_API_KEY"
_PUBLIC_PATHS = frozenset({"/health"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_MAX_BODY_BYTES = 1 << 20


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: Literal["ok"] = "ok"
    service: str = "fx1-harness-api"
    version: str = __version__
    registered_commands: int
    backends: dict[str, bool]


class HarnessCommandItem(_Model):
    name: str
    role: str
    argv: list[str]
    timeout_s: int
    description: str


class HarnessCommandListResponse(_Model):
    items: list[HarnessCommandItem]
    total: int


class HarnessRunRequest(_Model):
    command: str
    extra_args: list[str] = Field(default_factory=list, max_length=64)
    config: str | None = None


class HarnessRunResponse(_Model):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    ok: bool
    timeout_s: int


class ChatMessage(_Model):
    role: str
    content: str


class CompleteRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    checkpoint_dir: str | None = None
    receipt_hashes: list[str] | None = None


class CompleteResponse(_Model):
    backend: str
    model: str | None
    content: str
    receipt_hashes: list[str]


class ReceiptVerifyRequest(_Model):
    receipt: dict[str, Any]


class ReceiptVerifyResponse(_Model):
    valid: bool
    path: str
    schema_tag: str
    kind: str | None
    verdict: str | None
    digest_convention: str | None
    errors: list[str]
    warnings: list[str]


def _backend_configured() -> dict[str, bool]:
    """Presence-of-credentials flags only — values never leave the process."""
    checkpoint_env = os.environ.get("FX1_CHECKPOINT_DIR", "")
    return {
        "hosted_k3": bool(os.environ.get("MOONSHOT_API_KEY")),
        "byok": all(
            os.environ.get(name) for name in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV)
        ),
        "local_fx1": bool(checkpoint_env) and (Path(checkpoint_env) / "modelcard.json").is_file(),
    }


def create_app(
    harness: Harness | None = None,
    backend_resolver: Any | None = None,
) -> FastAPI:
    api_key = os.environ.get(_API_KEY_ENV) or None
    lab = harness or Harness()
    resolve_backend = backend_resolver or get_backend

    app = FastAPI(
        title="fx-1 harness API",
        version=__version__,
        description=(
            "Execution surface for the fx-1 harness: the registered lab "
            "commands, sealed-receipt verification, and gated model "
            "completion over hosted_k3 / local_fx1 / BYOK backends."
        ),
    )

    @app.middleware("http")
    async def harness_api_auth(request: Request, call_next: Any) -> Any:
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError:
                    return JSONResponse(
                        status_code=400, content={"detail": "invalid content-length"}
                    )
                if length > _MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"body exceeds {_MAX_BODY_BYTES}-byte cap"},
                    )
        if request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
        elif api_key:
            provided = request.headers.get("X-API-Key")
            if not provided or not hmac.compare_digest(provided, api_key):
                response = JSONResponse(
                    status_code=401, content={"detail": "invalid or missing X-API-Key"}
                )
            else:
                response = await call_next(request)
        else:
            host = (request.client.host if request.client else "") or ""
            if host not in _LOOPBACK_HOSTS:
                response = JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "FX1_API_KEY is unset; non-localhost clients are "
                            "refused. Set FX1_API_KEY and send X-API-Key, or "
                            "bind to 127.0.0.1 only."
                        )
                    },
                )
            else:
                response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            registered_commands=len(lab.list_commands()),
            backends=_backend_configured(),
        )

    @app.get("/harness/commands", response_model=HarnessCommandListResponse)
    def list_commands(
        role: HarnessRole | None = Query(default=None),
    ) -> HarnessCommandListResponse:
        items = [
            HarnessCommandItem(
                name=c.name,
                role=str(c.role),
                argv=list(c.argv),
                timeout_s=c.timeout_s,
                description=c.description,
            )
            for c in lab.list_commands(role)
        ]
        return HarnessCommandListResponse(items=items, total=len(items))

    @app.post("/harness/runs", response_model=HarnessRunResponse)
    def run_command(body: HarnessRunRequest) -> HarnessRunResponse:
        try:
            result = lab.run(
                body.command,
                body.extra_args or None,
                config=Path(body.config) if body.config else None,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        command = lab.get(body.command)
        return HarnessRunResponse(
            command=result.command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            ok=result.ok,
            timeout_s=command.timeout_s,
        )

    @app.post("/harness/complete", response_model=CompleteResponse)
    def complete(body: CompleteRequest) -> CompleteResponse:
        kwargs: dict[str, Any] = {}
        if body.backend == "local_fx1":
            checkpoint = body.checkpoint_dir or os.environ.get("FX1_CHECKPOINT_DIR")
            if not checkpoint:
                raise HTTPException(
                    422,
                    "local_fx1 needs a checkpoint_dir in the request or "
                    "FX1_CHECKPOINT_DIR on the server",
                )
            kwargs["checkpoint_dir"] = checkpoint
        elif body.checkpoint_dir is not None:
            raise HTTPException(422, "checkpoint_dir applies only to the local_fx1 backend")
        try:
            backend = resolve_backend(body.backend, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            # Missing credentials / unsigned release / failed ship gate are
            # server-side configuration faults, not client input errors.
            raise HTTPException(503, str(exc)) from exc
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        try:
            content = cited_complete(backend, messages, receipt_hashes=body.receipt_hashes)
        except NotImplementedError as exc:
            raise HTTPException(501, str(exc)) from exc
        except Fx1HonestyError as exc:
            # The model produced a contract-violating headline; the gate
            # caught it before the bytes left — surface as 502, not success.
            raise HTTPException(502, f"honesty gate refused model output: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        model_name = getattr(backend, "_model", None)
        return CompleteResponse(
            backend=body.backend,
            model=model_name if isinstance(model_name, str) else None,
            content=content,
            receipt_hashes=body.receipt_hashes or [],
        )

    @app.post("/receipts/verify", response_model=ReceiptVerifyResponse)
    def verify_receipt(body: ReceiptVerifyRequest) -> ReceiptVerifyResponse:
        result = verify_receipt_payload(body.receipt, path=Path("<api>"))
        return ReceiptVerifyResponse(
            valid=result["valid"],
            path=result["path"],
            schema_tag=result["schema"],
            kind=result["kind"] if isinstance(result["kind"], str) else None,
            verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
            digest_convention=result["digest_convention"],
            errors=list(result["errors"]),
            warnings=list(result["warnings"]),
        )

    return app


app = create_app()


if __name__ == "__main__":
    # `python -m fx1.serve.api` — loopback guard matches the research API.
    import uvicorn

    host = os.environ.get("FX1_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FX1_API_PORT", "8011"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get(_API_KEY_ENV):
        raise SystemExit(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure"
        )
    uvicorn.run(app, host=host, port=port, reload=False)
