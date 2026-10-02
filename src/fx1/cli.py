"""``fx1`` command-line interface — the model project's front door.

dipcatcher remains available as the harness CLI (``dipcatcher``/``quant``
aliases); this CLI drives the fx-1 lifecycle: corpus, eval, training
manifests, and harness inspection.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

if TYPE_CHECKING:
    from fx1.eval.suite import ModelFn
    from fx1.sdk import Fx1Harness
    from fx1.serve.client import HarnessClient

_REMOTE_HELP = (
    "Drive a remote harness API at this base URL (HarnessClient) instead of the in-process SDK."
)
_API_KEY_HELP = "X-API-Key for the remote harness; falls back to FX1_API_KEY."
_TIMEOUT_HELP = "Remote request timeout in seconds."
_BYOK_URL_HELP = "Per-request BYOK endpoint (OpenAI-compatible base URL)."
_BYOK_KEY_HELP = "Per-request BYOK API key."
_BYOK_MODEL_HELP = "Per-request BYOK model name."


def _byok_opts(
    base_url: str | None, api_key: str | None, model: str | None
) -> dict[str, str] | None:
    """Pack the three --byok-* flags into the wire override; all-or-none."""
    parts = {"base_url": base_url, "api_key": api_key, "model": model}
    if all(v is None for v in parts.values()):
        return None
    if any(v is None for v in parts.values()):
        typer.echo("--byok-base-url, --byok-api-key and --byok-model go together", err=True)
        raise typer.Exit(2)
    return {k: str(v) for k, v in parts.items()}


def _surface(
    remote: str | None, api_key: str | None, timeout_s: float
) -> Fx1Harness | HarnessClient:
    """In-process SDK by default; HarnessClient when ``--remote`` is given."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        return Fx1Harness()
    from fx1.serve.client import HarnessClient

    return HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )


def _or_exit[T](fn: Callable[[], T]) -> T:
    """Run a surface call; faults print one clean line and exit 2."""
    try:
        return fn()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 — CLI reports the class+message, not a traceback
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


app = typer.Typer(
    name="fx1",
    help="fx-1 — the quant LLM. dipcatcher is the harness that builds, evaluates, and verifies it.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Show the fx-1 package version and exit.", is_eager=True
    ),
) -> None:
    if version:
        from fx1 import __version__

        typer.echo(f"fx1 {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


corpus_app = typer.Typer(help="Training-corpus construction.")
train_app = typer.Typer(help="Training-run manifests (LoRA/QLoRA on K3).")
harness_app = typer.Typer(help="Inspect/run the dipcatcher harness.")
sources_app = typer.Typer(help="Professional datasource registry, routing, fetch.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(train_app, name="train")
app.add_typer(harness_app, name="harness")
app.add_typer(sources_app, name="sources")


@harness_app.command("operations")
def harness_operations(
    query: str = typer.Argument("", help="Substring filter over operation ids."),
    kind: str | None = typer.Option(None, help="Restrict to one kind: feature, skill, or plugin."),
    offset: int = typer.Option(0, min=0, help="Zero-based page offset."),
    limit: int = typer.Option(20, min=1, max=100, help="Page size (1-100)."),
) -> None:
    """List registered dipcatcher capabilities through the explicit registry."""
    from fx1.operations.base import OperationKind
    from fx1.operations.registry import list_operations

    if kind is not None and kind not in ("feature", "skill", "plugin"):
        raise typer.BadParameter("must be feature, skill, or plugin", param_hint="--kind")
    kind_arg = cast(OperationKind, kind) if kind is not None else None
    typer.echo(
        json.dumps(
            list_operations(query, kind=kind_arg, offset=offset, limit=limit),
            indent=2,
        )
    )


@harness_app.command("describe-operation")
def harness_describe_operation(
    operation_id: str = typer.Argument(
        ..., help="Registered operation id, e.g. features.simple_returns."
    ),
) -> None:
    """Print the exact input/output schema of one registered operation."""
    from fx1.operations.registry import get_operation

    typer.echo(json.dumps(get_operation(operation_id).describe(), indent=2))


_OPERATION_ARGUMENT_MAX_BYTES = 2_000_000


@harness_app.command("execute-operation")
def harness_execute_operation(
    operation_id: str = typer.Argument(..., help="Registered operation id."),
    arguments: str | None = typer.Option(
        None, "--arguments", help="JSON object of inputs (at most 2,000,000 UTF-8 bytes)."
    ),
    arguments_file: Path | None = typer.Option(
        None,
        "--arguments-file",
        help="Path to a UTF-8 JSON file of inputs (at most 2,000,000 bytes).",
    ),
    workspace_root: Path | None = typer.Option(
        None, "--workspace-root", help="Workspace filesystem boundary."
    ),
) -> None:
    """Execute one registered operation with validated, bounded inputs."""
    if (arguments is None) == (arguments_file is None):
        raise typer.BadParameter("Provide exactly one of --arguments or --arguments-file.")
    try:
        if arguments is not None:
            # Check characters first so even encoding a large argv value is bounded.
            if len(arguments) > _OPERATION_ARGUMENT_MAX_BYTES:
                raise ValueError(
                    f"arguments exceed the {_OPERATION_ARGUMENT_MAX_BYTES}-byte budget"
                )
            raw = arguments.encode("utf-8")
        else:
            assert arguments_file is not None
            # Read one sentinel byte beyond the budget; never materialize the whole file.
            with arguments_file.open("rb") as stream:
                raw = stream.read(_OPERATION_ARGUMENT_MAX_BYTES + 1)
        if len(raw) > _OPERATION_ARGUMENT_MAX_BYTES:
            raise ValueError(f"arguments exceed the {_OPERATION_ARGUMENT_MAX_BYTES}-byte budget")
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, OSError, RecursionError) as exc:
        raise typer.BadParameter(f"cannot read arguments: {exc}") from exc
    try:
        resolved_root = workspace_root.resolve() if workspace_root is not None else None
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(
            f"cannot resolve workspace root: {exc}", param_hint="--workspace-root"
        ) from exc
    from fx1.operations.registry import execute_operation

    result = execute_operation(operation_id, payload, workspace_root=resolved_root)
    typer.echo(json.dumps(result, indent=2))


@corpus_app.command("build")
def corpus_build(
    receipts_dir: list[Path] = typer.Option(
        [Path("receipts")],
        help="Receipt directory (repeatable — the flywheel spans several).",
    ),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl"), help="Output JSONL."),
) -> None:
    """Build the fx-1 SFT corpus from gate-passed dipcatcher receipts."""
    from fx1.data import build_corpus

    stats = build_corpus(list(receipts_dir), out)
    typer.echo(json.dumps(stats, indent=2))


@corpus_app.command("build-full")
def corpus_build_full(
    receipts_dir: list[Path] = typer.Option([Path("receipts")]),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    notebooks: list[Path] = typer.Option([], help="Research notebooks/docs to include."),
    artifacts_dir: Path | None = typer.Option(None, help="Ledger artifacts dir."),
) -> None:
    """Full corpus: receipts + notebooks + ledgers, all provenance-hashed."""
    from fx1.data import build_full_corpus

    stats = build_full_corpus(
        list(receipts_dir), out, notebooks=list(notebooks), artifacts_dir=artifacts_dir
    )
    typer.echo(json.dumps(stats, indent=2))


@train_app.command("manifest")
def train_manifest(
    config: Path = typer.Option(..., help="TrainConfig JSON file."),
    out: Path = typer.Option(Path("data/fx1/manifest.json")),
) -> None:
    """Validate the run contract (eval-before-train, provenance, cost) and
    write an immutable training manifest."""
    from fx1.train import TrainConfig, build_training_manifest

    cfg = TrainConfig.model_validate_json(config.read_text(encoding="utf-8"))
    manifest = build_training_manifest(cfg, out)
    typer.echo(json.dumps({"run_name": manifest["run_name"], "out": str(out)}))


@harness_app.command("list")
def harness_list(
    role: str | None = typer.Option(
        None, help="Filter: data_engine | evaluation | verification | model_training"
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List the lab commands fx-1 may invoke through the harness."""
    if remote is not None:
        from fx1.serve.client import HarnessClient

        surface = HarnessClient(
            remote,
            api_key=api_key or os.environ.get("FX1_API_KEY") or None,
            timeout_s=timeout_s,
        )
        for name in _or_exit(lambda: surface.commands(role=role)):
            typer.echo(name)
        return
    from fx1.harness import Harness, HarnessRole

    role_filter = HarnessRole(role) if role else None
    for cmd in Harness().list_commands(role=role_filter):
        typer.echo(f"{cmd.name:<18} [{cmd.role}] {cmd.description}")


@harness_app.command("run")
def harness_run(
    name: str = typer.Argument(..., help="Registered harness command name."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        help="Dedup key for the remote submission; a retried call returns the stored result. Auto-minted per invocation.",
    ),
) -> None:
    """Run a registered dipcatcher harness command (fail-closed registry)."""
    if remote is not None:
        from fx1.serve.client import HarnessClient

        client = HarnessClient(
            remote,
            api_key=api_key or os.environ.get("FX1_API_KEY") or None,
            timeout_s=timeout_s,
        )
        result = _or_exit(lambda: client.run(name, idempotency_key=idempotency_key))
    else:
        from fx1.harness import Harness

        result = Harness().run(name)
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    raise typer.Exit(code=result.exit_code)


@harness_app.command("serve")
def harness_serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8011, help="Bind port."),
    max_inflight: int | None = typer.Option(
        None, help="Concurrent heavy requests (env FX1_API_MAX_INFLIGHT, default 16)."
    ),
    job_max: int | None = typer.Option(
        None, help="Job-store capacity (env FX1_API_JOB_MAX, default 1024)."
    ),
    idem_max: int | None = typer.Option(
        None, help="Idempotency-store capacity (env FX1_API_IDEM_MAX, default 1024)."
    ),
    sse_keepalive_s: float | None = typer.Option(
        None,
        help="SSE keepalive interval seconds (env FX1_API_SSE_KEEPALIVE_S, default 15).",
    ),
    rate_limit_rps: float | None = typer.Option(
        None, help="Per-client req/s cap (env FX1_API_RATE_LIMIT_RPS, 0 = off)."
    ),
    gzip_min_bytes: int | None = typer.Option(
        None,
        help="Response compression floor bytes (env FX1_API_GZIP_MIN_BYTES, 0 = off).",
    ),
    cors_origins: str | None = typer.Option(
        None,
        help="Comma-separated allowed browser origins (env FX1_API_CORS_ORIGINS, empty = off).",
    ),
    breaker_threshold: int | None = typer.Option(
        None,
        help="Consecutive backend faults that open the circuit (env FX1_API_BREAKER_THRESHOLD, 0 = off).",
    ),
    breaker_cooldown_s: float | None = typer.Option(
        None,
        help="Seconds an open circuit fast-fails before a probe (env FX1_API_BREAKER_COOLDOWN_S).",
    ),
    receipts_dir: str | None = typer.Option(
        None,
        help="Sealed-receipts directory for GET /receipts fetch (env FX1_API_RECEIPTS_DIR).",
    ),
) -> None:
    """Serve the harness API (POST /harness/runs, /harness/complete, /receipts/verify)."""
    import uvicorn

    from fx1.serve.api import create_app

    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get("FX1_API_KEY"):
        typer.echo(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        harness_api = create_app(
            max_inflight=max_inflight,
            sse_keepalive_s=sse_keepalive_s,
            idem_max=idem_max,
            job_max=job_max,
            rate_limit_rps=rate_limit_rps,
            gzip_min_bytes=gzip_min_bytes,
            cors_origins=cors_origins,
            breaker_threshold=breaker_threshold,
            breaker_cooldown_s=breaker_cooldown_s,
            receipts_dir=receipts_dir,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    uvicorn.run(harness_api, host=host, port=port, reload=False, server_header=False)


_BACKEND_HELP = (
    "hosted_k3 | local_fx1 | byok (BYOK reads FX1_BYOK_BASE_URL / "
    "FX1_BYOK_API_KEY / FX1_BYOK_MODEL)"
)


@harness_app.command("complete")
def harness_complete(
    prompt: str = typer.Argument(..., help="User message to complete."),
    backend: str = typer.Option("local_fx1", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    receipt: list[str] = typer.Option([], "--receipt", help="Receipt sha256 to cite (repeatable)."),
    stream: bool = typer.Option(
        False, "--stream", help="Emit gated token deltas instead of one block."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
) -> None:
    """One gated completion — the honesty gate runs before output."""
    surface = _surface(remote, api_key, timeout_s)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    if stream:
        chunks = _or_exit(
            lambda: surface.stream_complete(
                [{"role": "user", "content": prompt}],
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                receipt_hashes=receipt or None,
                byok=byok,
                timeout_s=backend_timeout,
            )
        )
        for chunk in chunks:
            typer.echo(chunk, nl=False)
        typer.echo()
        return
    out = _or_exit(
        lambda: surface.complete(
            [{"role": "user", "content": prompt}],
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            receipt_hashes=receipt or None,
            byok=byok,
            timeout_s=backend_timeout,
        )
    )
    typer.echo(out.content)


@harness_app.command("batch")
def harness_batch(
    prompts_file: Path = typer.Argument(
        ..., help="JSON array or JSONL of prompt strings to complete."
    ),
    backend: str = typer.Option("local_fx1", help=_BACKEND_HELP),
    workers: int = typer.Option(4, "--workers", help="Concurrent slots (1-16)."),
    receipt: list[str] = typer.Option([], "--receipt", help="Receipt sha256 to cite (repeatable)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    out_file: Path | None = typer.Option(
        None, "--out", help="Write the JSON results here (default: stdout)."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """Gated batch completion — per-item failures surface as exit 2."""
    surface = _surface(remote, api_key, timeout_s)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    prompts = _or_exit(lambda: _load_prompts(prompts_file))
    results = _or_exit(
        lambda: surface.complete_many(
            [[{"role": "user", "content": p}] for p in prompts],
            backend=backend,
            receipt_hashes=receipt or None,
            byok=byok,
            timeout_s=backend_timeout,
            max_workers=workers,
        )
    )
    payload = json.dumps(
        [{"prompt": p, "content": r.content} for p, r in zip(prompts, results, strict=True)],
        indent=2,
    )
    if out_file is not None:
        out_file.write_text(payload + "\n", encoding="utf-8")
    else:
        typer.echo(payload)


def _load_prompts(path: Path) -> list[str]:
    """JSON array of strings or JSONL of str / ``{\"prompt\": ...}`` lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path}: empty prompts file")
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    prompts = [it["prompt"] if isinstance(it, dict) else it for it in items]
    if not all(isinstance(p, str) and p for p in prompts):
        raise ValueError(f"{path}: every prompt must be a non-empty string")
    return prompts


@harness_app.command("verify")
def harness_verify(
    receipt_path: Path = typer.Argument(
        ..., help="Receipt JSON file — or a directory of them — to verify."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Deep-verify receipt file(s) through the verifier surface."""
    surface = _surface(remote, api_key, timeout_s)
    if receipt_path.is_dir():
        files = sorted(receipt_path.glob("*.json"))
        if not files:
            typer.echo(f"error: no *.json receipts under {receipt_path}", err=True)
            raise typer.Exit(code=2)

        def _verify_one(f: Path) -> dict[str, Any]:
            v = _or_exit(lambda: surface.verify_receipt(json.loads(f.read_text())))
            return {"file": f.name, "valid": v.valid}

        results = [_verify_one(f) for f in files]
        typer.echo(
            json.dumps(
                {
                    "files": len(results),
                    "valid": sum(1 for r in results if r["valid"]),
                    "results": results,
                },
                indent=2,
            )
        )
        raise typer.Exit(code=0 if all(r["valid"] for r in results) else 1)
    verdict = _or_exit(lambda: surface.verify_receipt(json.loads(receipt_path.read_text())))
    typer.echo(
        json.dumps(
            {
                "valid": verdict.valid,
                "errors": list(verdict.errors),
                "warnings": list(verdict.warnings),
                "schema": verdict.schema_tag,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if verdict.valid else 1)


@harness_app.command("receipts")
def harness_receipts(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    receipts_dir: Path = typer.Option(
        Path("receipts"),
        "--receipts-dir",
        help="Local sealed-receipts store (ignored with --remote).",
    ),
) -> None:
    """Index the sealed-receipt store — a local dir or the remote's over HTTP."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        surface: Fx1Harness | HarnessClient = Fx1Harness(receipts_dir=receipts_dir)
    else:
        surface = _remote_client(remote, api_key, timeout_s)
    items = _or_exit(lambda: surface.receipts())
    typer.echo(
        json.dumps(
            {
                "count": len(items),
                "items": [{"sha256": r.sha256, "name": r.name} for r in items],
            },
            indent=2,
        )
    )


@harness_app.command("receipt")
def harness_receipt(
    sha256: str = typer.Argument(..., help="Receipt content hash (64 lowercase hex)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    receipts_dir: Path = typer.Option(
        Path("receipts"),
        "--receipts-dir",
        help="Local sealed-receipts store (ignored with --remote).",
    ),
) -> None:
    """Fetch one sealed receipt by content hash: verbatim document + validity."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        surface: Fx1Harness | HarnessClient = Fx1Harness(receipts_dir=receipts_dir)
    else:
        surface = _remote_client(remote, api_key, timeout_s)
    r = _or_exit(lambda: surface.receipt(sha256))
    typer.echo(json.dumps({"sha256": r.sha256, "valid": r.valid, "receipt": r.document}, indent=2))


@harness_app.command("probe")
def harness_probe(
    backend: str = typer.Option("byok", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    prompt: str = typer.Option("ping", "--prompt", help="Probe prompt."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds (route default: 30)."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
) -> None:
    """Deep health: one live gated completion per call — prints the
    verdict JSON and exits 0 when ok, 1 when the backend is unhealthy.
    ``--remote`` probes via the wire route (probe verdicts never feed
    the remote's circuit breaker); in-process probes run the same
    resolver the SDK uses, so a BYOK probe tests your own endpoint."""
    surface = _surface(remote, api_key, timeout_s)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    out = _or_exit(
        lambda: surface.probe_backend(
            backend,
            checkpoint_dir=checkpoint_dir,
            byok=byok,
            timeout_s=backend_timeout,
            prompt=prompt,
        )
    )
    typer.echo(
        json.dumps(
            {
                "backend": out.backend,
                "ok": out.ok,
                "model": out.model,
                "latency_ms": round(out.latency_ms, 1),
                "error": out.error,
                "error_class": out.error_class,
            },
            indent=2,
        )
    )
    if not out.ok:
        raise typer.Exit(code=1)


@harness_app.command("check-text")
def harness_check_text(
    text: str = typer.Argument(..., help="Text to run through the honesty gate."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Pre-flight text through the honesty gate — prints the verdict JSON
    and exits 0 when clean / 1 on refusal. In-process (no backend needed);
    ``--remote`` checks via the wire route."""
    surface = _surface(remote, api_key, timeout_s)
    out = _or_exit(lambda: surface.check_text(text))
    typer.echo(json.dumps({"ok": out.ok, "error": out.error}, indent=2))
    if not out.ok:
        raise typer.Exit(code=1)


def _record_json(rec: Any) -> dict[str, Any]:
    return {
        "completion_id": rec.completion_id,
        "backend": rec.backend,
        "model": rec.model,
        "ok": rec.ok,
        "latency_ms": rec.latency_ms,
        "at": rec.at,
        "usage": rec.usage,
        "error": rec.error,
        "error_class": rec.error_class,
        "prompt_sha256": rec.prompt_sha256,
        "output_sha256": rec.output_sha256,
    }


@harness_app.command("completions")
def harness_completions(
    limit: int = typer.Option(50, "--limit", help="Newest N records (log is ring-bounded)."),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Newest-first window on the completion log — per-call evidence
    (hashes, usage, verdict) for every gated call this surface served."""
    surface = _surface(remote, api_key, timeout_s)
    items = _or_exit(lambda: surface.completions(limit=limit, backend=backend))
    typer.echo(
        json.dumps({"count": len(items), "items": [_record_json(r) for r in items]}, indent=2)
    )


@harness_app.command("completion")
def harness_completion(
    completion_id: str = typer.Argument(..., help="Completion record id (hex)."),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_completion_record.v1 doc instead."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch one recorded call by id — the audit handle minted per call.
    ``--receipt`` prints the sealed export instead (verify with
    ``verify-research`` or ``POST /receipts/verify``)."""
    surface = _surface(remote, api_key, timeout_s)
    if receipt:
        doc = _or_exit(lambda: surface.completion_receipt(completion_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    rec = _or_exit(lambda: surface.completion(completion_id))
    typer.echo(json.dumps(_record_json(rec), indent=2))


@harness_app.command("health")
def harness_health(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Configured-backend presence booleans — never secret values."""
    surface = _surface(remote, api_key, timeout_s)
    h = _or_exit(surface.health)
    typer.echo(
        json.dumps(
            {
                "status": h.status,
                "version": h.version,
                "registered_commands": h.registered_commands,
                "backends": h.backends,
            },
            indent=2,
        )
    )


@harness_app.command("metrics")
def harness_metrics(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    format: str = typer.Option(
        "json", "--format", help="Output format: 'json' or 'prom' (Prometheus text)"
    ),
) -> None:
    """Remote ops counters — requires --remote (nothing meters in-process)."""
    if remote is None:
        typer.echo(
            "error: harness metrics is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    if format not in ("json", "prom"):
        typer.echo(f"error: --format must be 'json' or 'prom', got {format!r}", err=True)
        raise typer.Exit(code=2)
    from fx1.serve.client import HarnessClient

    client = HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )
    if format == "prom":
        typer.echo(_or_exit(client.metrics_text))
        return
    m = _or_exit(client.metrics)
    typer.echo(
        json.dumps(
            {
                "uptime_s": m.uptime_s,
                "requests_total": m.requests_total,
                "errors_total": m.errors_total,
                "by_status": m.by_status,
                "inflight": m.inflight,
                "inflight_watermark": m.inflight_watermark,
                "max_inflight": m.max_inflight,
                "draining": m.draining,
                "rate_limited_total": m.rate_limited_total,
                "complete": m.complete,
            },
            indent=2,
        )
    )


@harness_app.command("drain")
def harness_drain(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    wait_s: float = typer.Option(
        0.0,
        "--wait-s",
        help="Block server-side up to N seconds for in-flight work to drain.",
    ),
) -> None:
    """Latch the remote harness into drain mode — one-way, idempotent.

    Gated routes start refusing new work with 503 while in-flight
    requests finish; ``--wait-s N`` blocks until the pool empties and
    reports ``drained``; otherwise watch ``harness metrics --remote``
    until ``inflight`` reaches zero, then stop the process.
    """
    if remote is None:
        typer.echo(
            "error: harness drain is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve.client import HarnessClient

    client = HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )
    typer.echo(json.dumps(_or_exit(lambda: client.drain(wait_s=wait_s)), indent=2))


@harness_app.command("version")
def harness_version(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Wire-contract + package versions — remote mode asks the server's
    /harness/version, local mode prints this install's own pair."""
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        typer.echo(json.dumps(_or_exit(lambda: client.server_version())))
        return
    from fx1 import __version__
    from fx1.serve.contract import API_VERSION

    typer.echo(json.dumps({"api_version": API_VERSION, "fx1_version": __version__, "local": True}))


@harness_app.command("compat")
def harness_compat(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Wire-contract negotiation: prints the compat report and exits 0 when
    the remote speaks this client's api_version, 1 on any mismatch (incl.
    a peer too old to have a version route)."""
    if remote is None:
        typer.echo(
            "error: harness compat is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    client = _remote_client(remote, api_key, timeout_s)
    report = _or_exit(lambda: client.check_compat(strict=False))
    typer.echo(json.dumps(report))
    if not report["compatible"]:
        raise typer.Exit(code=1)


@harness_app.command("capabilities")
def harness_capabilities(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Server self-description: wire features + effective limits, so a
    deploy can sanity-check the remote's config before routing work."""
    if remote is None:
        typer.echo(
            "error: harness capabilities is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    client = _remote_client(remote, api_key, timeout_s)
    typer.echo(json.dumps(_or_exit(lambda: client.capabilities())))


@harness_app.command("ready")
def harness_ready(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Readiness probe: exits 0 while the remote accepts work, 1 once
    draining — for deploy scripts and Kubernetes readiness gates."""
    if remote is None:
        typer.echo(
            "error: harness ready is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve.backends import BackendNotConfiguredError
    from fx1.serve.client import HarnessClient

    client = HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )
    try:
        out = client.ready()
    except BackendNotConfiguredError:
        typer.echo(json.dumps({"ready": False}))
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(out))


def _remote_client(remote: str, api_key: str | None, timeout_s: float) -> HarnessClient:
    from fx1.serve.client import HarnessClient

    return HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )


def _need_remote(remote: str | None) -> None:
    if remote is None:
        typer.echo(
            "error: async jobs are a wire surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)


@harness_app.command("submit")
def harness_submit(
    name: str = typer.Argument(..., help="Registered harness command name."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        help="Dedup key for the submission; a retried submit returns the same job id.",
    ),
    callback_url: str | None = typer.Option(
        None,
        "--callback-url",
        help="http(s) webhook; the full job record is POSTed on terminal status.",
    ),
    callback_secret: str | None = typer.Option(
        None,
        "--callback-secret",
        help="HMAC secret signing the webhook (X-Fx1-Webhook-Signature).",
    ),
) -> None:
    """Submit a run as a background job; prints the job id."""
    _need_remote(remote)
    job_id = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).submit_run(
            name,
            idempotency_key=idempotency_key,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    typer.echo(job_id)


@harness_app.command("submit-batch")
def harness_submit_batch(
    spec: Path = typer.Argument(
        ..., help="JSON file: a list of run-request objects (command, extra_args, …)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Submit a batch of jobs in one request; prints the per-item outcome JSON."""
    _need_remote(remote)
    try:
        jobs = json.loads(spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"cannot read batch spec {spec}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(jobs, list):
        typer.echo("batch spec must be a JSON list of run-request objects", err=True)
        raise typer.Exit(code=2)
    out = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).submit_batch(jobs))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("job")
def harness_job(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_job_record.v1 doc instead."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a job's live status record. ``--receipt`` prints the sealed
    export instead (verify with ``verify-research`` /
    ``POST /receipts/verify``)."""
    _need_remote(remote)
    if receipt:
        doc = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).job_receipt(job_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).job_status(job_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("jobs")
def harness_jobs(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    status: str | None = typer.Option(
        None, "--status", help="Filter: queued|running|succeeded|failed|cancelled."
    ),
    limit: int = typer.Option(100, "--limit", help="Page size (max 500)."),
    offset: int = typer.Option(0, "--offset", help="Skip the newest N jobs."),
) -> None:
    """List the job inventory (newest first) with total for paging."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).list_jobs(
            status=status, limit=limit, offset=offset
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("cancel")
def harness_cancel(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a queued job; running/terminal jobs report a 409 conflict."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).cancel_job(job_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("wait")
def harness_wait(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help="Status poll interval, seconds."),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (job keeps running)."
    ),
) -> None:
    """Wait for a job to finish, then print its stdout like harness run."""
    _need_remote(remote)
    result = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_run(
            job_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    raise typer.Exit(code=result.exit_code)


@harness_app.command("watch")
def harness_watch(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    stream_timeout_s: float = typer.Option(
        600.0, "--stream-timeout", help="Server-side SSE stream bound, seconds."
    ),
) -> None:
    """Follow a job over the SSE event stream — one connection, no polling."""
    _need_remote(remote)
    frames = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).stream_job(
            job_id, timeout_s=stream_timeout_s
        )
    )
    for st in frames:
        typer.echo(f"{st['status']}\t{job_id}")
    last = frames[-1]
    if last["status"] == "succeeded":
        result = last["result"]
        typer.echo(result["stdout"])
        if result["stderr"]:
            typer.echo(result["stderr"], err=True)
        raise typer.Exit(code=result["exit_code"])
    if last["status"] == "failed":
        typer.echo(f"job failed: {last.get('error')}", err=True)
        raise typer.Exit(code=1)
    if last["status"] == "cancelled":
        typer.echo("job cancelled", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"stream ended before terminal state (last status {last['status']!r})", err=True)
    raise typer.Exit(code=1)


@app.command("eval")
def eval_bank(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    out: Path = typer.Option(Path("data/fx1/eval.json")),
) -> None:
    """Run the built-in eval task bank against an fx-1 backend."""
    from fx1.eval import DEFAULT_BANK, run_suite

    model = _resolve_model_backend(backend, checkpoint_dir)
    summary = run_suite(model, list(DEFAULT_BANK))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"by_kind": summary["by_kind"], "honesty_gate_passed": summary["honesty_gate_passed"]},
            indent=2,
        )
    )


def _resolve_model_backend(backend: str, checkpoint_dir: Path | None) -> ModelFn:
    """Resolve the model under test. Fail-closed on unknown names.

    ``local_fx1`` needs ``--checkpoint-dir``; ``byok`` reads the
    FX1_BYOK_* env contract (never argv); ``hosted_k3`` needs
    MOONSHOT_API_KEY. An unknown backend string exits 2 — silently
    running a different backend than the flag names would corrupt the
    receipt's meaning.
    """
    from fx1.serve import get_backend

    if backend == "local_fx1":
        return get_backend("local_fx1", checkpoint_dir=checkpoint_dir).complete
    if backend in ("hosted_k3", "byok"):
        return get_backend(backend).complete
    typer.echo(f"unknown --backend {backend!r}; choose hosted_k3 | local_fx1 | byok", err=True)
    raise typer.Exit(code=2)


_JUDGE_BACKEND_HELP = (
    "Optional 'hosted_k3' LLM judge for MT-Bench-style grading (needs "
    "MOONSHOT_API_KEY); omit for the deterministic rule-based judge."
)


def _resolve_judge(judge_backend: str | None) -> ModelFn | None:
    """Build the optional ext-bench judge ModelFn. Fail-closed on usage.

    ``None`` selects the deterministic rule-based judge by configuration;
    ``hosted_k3`` reuses the same backend construction as the model under
    test (a missing MOONSHOT_API_KEY raises, never fabricates a judge).
    """
    if judge_backend is None:
        return None
    if judge_backend not in ("hosted_k3", "byok"):
        typer.echo(
            f"unknown --judge-backend {judge_backend!r}; 'hosted_k3' and "
            "'byok' are supported (omit the flag for the deterministic "
            "rule-based judge)",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve import get_backend

    return get_backend(judge_backend).complete


@app.command("capability-eval")
def capability_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    judge_backend: str | None = typer.Option(None, "--judge-backend", help=_JUDGE_BACKEND_HELP),
    out: Path = typer.Option(Path("data/fx1/capability_eval.json")),
) -> None:
    """Run the capability battery: time-series reasoning, probability
    calibration, harness tool-use, retrieval-with-citation, external-
    benchmark-format adapters (MT-Bench / FinanceBench / FinToolBench-style),
    and options reasoning — all on seeded SYNTHETIC banks. Exit 1 when any
    honesty sub-gate, the calibration gate, or an ext-bench score gate
    fails. Real ext-bench JSONL sources plug in via `fx1 ext-bench-eval`;
    the options bank is sealed (no external loader)."""
    from fx1.eval import run_capability_eval

    judge = _resolve_judge(judge_backend)
    model = _resolve_model_backend(backend, checkpoint_dir)
    report = run_capability_eval(model, seed=seed, judge=judge)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "ts_reasoning_overall": report.ts_reasoning.overall,
                "calibration_ece": report.calibration.ece,
                "calibration_passed": report.calibration.passed,
                "tooluse_pass_rate": report.tooluse.pass_rate,
                "retrieval_accuracy": report.retrieval.accuracy,
                "ext_bench_honesty_gate_passed": (
                    report.ext_bench.honesty_gate_passed if report.ext_bench is not None else None
                ),
                "ext_bench_score_gate_passed": (
                    report.ext_bench.score_gate_passed if report.ext_bench is not None else None
                ),
                "options_reasoning_passed": (
                    report.options_reasoning.passed
                    if report.options_reasoning is not None
                    else None
                ),
                "honesty_gate_passed": report.honesty_gate_passed,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("ext-bench-eval")
def ext_bench_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    judge_backend: str | None = typer.Option(None, "--judge-backend", help=_JUDGE_BACKEND_HELP),
    mtbench_jsonl: Path | None = typer.Option(
        None,
        "--mtbench-jsonl",
        help="Genuine MT-Bench-style JSONL replacing the sealed synthetic bank.",
    ),
    financebench_jsonl: Path | None = typer.Option(
        None,
        "--financebench-jsonl",
        help="Genuine FinanceBench-style JSONL replacing the sealed synthetic bank.",
    ),
    fintoolbench_jsonl: Path | None = typer.Option(
        None,
        "--fintoolbench-jsonl",
        help="Genuine FinToolBench-style JSONL replacing the sealed synthetic bank.",
    ),
    out: Path = typer.Option(Path("data/fx1/ext_bench_eval.json")),
) -> None:
    """Run the external-benchmark-format adapters (MT-Bench / FinanceBench /
    FinToolBench-style). Default banks are sealed SYNTHETIC correctness
    gates — NOT market evidence and NOT real benchmark scores; genuine JSONL
    exports plug in per benchmark (schema-validated, fail-closed). Exit 1
    when any refusal/honesty gate or score gate fails."""
    from fx1.eval.ext_bench import run_ext_bench_eval

    judge = _resolve_judge(judge_backend)
    model = _resolve_model_backend(backend, checkpoint_dir)
    sources: dict[str, Path] = {}
    if mtbench_jsonl is not None:
        sources["mtbench"] = mtbench_jsonl
    if financebench_jsonl is not None:
        sources["financebench"] = financebench_jsonl
    if fintoolbench_jsonl is not None:
        sources["fintoolbench"] = fintoolbench_jsonl
    report = run_ext_bench_eval(model, seed=seed, judge=judge, sources=sources or None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "synthetic": report.synthetic,
                "label": report.label,
                "benchmarks": {
                    name: {
                        "n_instances": score.n_instances,
                        "metrics": score.metrics,
                        "refusal_gate_passed": score.refusal_gate_passed,
                        "honesty_violations": score.honesty_violations,
                        "gate_passed": score.gate_passed,
                    }
                    for name, score in report.benchmarks.items()
                },
                "honesty_gate_passed": report.honesty_gate_passed,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("options-reasoning-eval")
def options_reasoning_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    out: Path = typer.Option(Path("data/fx1/options_reasoning_eval.json")),
) -> None:
    """Run the sealed SYNTHETIC options-reasoning battery (LiveOption-
    inspired levels: action validity, decision quality, risk characteristics,
    outcome, bait refusals). Gold answers come from the repo's own pricing
    modules — correctness gates, NOT market evidence. The bank is sealed
    (no external JSONL loader exists), so there is no --options-jsonl
    pass-through. Exit 1 when the bait/honesty gate fails."""
    from fx1.eval.options_reasoning_eval import run_options_reasoning_eval

    model = _resolve_model_backend(backend, checkpoint_dir)
    report = run_options_reasoning_eval(model, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "label": report.label,
                "n_items": report.n_items,
                "overall": report.overall,
                "by_level": report.by_level,
                "bait_accuracy": report.bait_accuracy,
                "bait_gate_passed": report.bait_gate_passed,
                "honesty_violations": report.honesty_violations,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("modelcard")
def modelcard_validate(path: Path = typer.Argument(...)) -> None:
    """Validate an fx-1 model card and report ship-gate status."""
    from fx1.modelcard import ModelCard

    card = ModelCard.load(path)
    typer.echo(
        json.dumps(
            {"version": card.version, "ship_eligible": card.eval_delta.ship_eligible}, indent=2
        )
    )


@app.command("redteam")
def redteam(
    backend: str = typer.Option("hosted_k3"),
    checkpoint_dir: Path | None = typer.Option(None),
) -> None:
    """Run the adversarial red-team suite against an fx-1 backend."""
    from fx1.eval.redteam import REDTEAM_TASKS
    from fx1.eval.suite import run_suite

    model = _resolve_model_backend(backend, checkpoint_dir)
    summary = run_suite(model, list(REDTEAM_TASKS))
    failed = [r["task"] for r in summary.results if not r["passed"]]
    typer.echo(
        json.dumps(
            {"honesty_gate_passed": summary["honesty_gate_passed"], "results": failed}, indent=2
        )
    )
    raise typer.Exit(code=0 if summary["honesty_gate_passed"] else 1)


@app.command("dpo")
def dpo_build(
    out: Path = typer.Option(Path("data/fx1/dpo.jsonl")),
) -> None:
    """Build contract-derived DPO preference pairs."""
    from fx1.train.dpo import build_preference_pairs

    pairs = build_preference_pairs(out)
    typer.echo(json.dumps({"pairs": len(pairs), "out": str(out)}))


@app.command("curriculum")
def curriculum_build(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus_curriculum.jsonl")),
) -> None:
    """Order the corpus contracts -> interpretation -> loops -> refusal."""
    from fx1.train.curriculum import build_curriculum

    counts = build_curriculum(corpus, out)
    typer.echo(json.dumps(counts, indent=2))


@app.command("maskedaEval")
def masked_eval(
    budget: float = typer.Option(0.25, help="Memory-gap ship budget."),
) -> None:
    """Masked/unmasked twin evaluation + memory-gap ship metric (offline
    structural check; live model runs inject a backend via fx1.eval)."""
    from fx1.eval import DEFAULT_BANK, masked_twins

    twins = masked_twins(list(DEFAULT_BANK))
    typer.echo(
        json.dumps(
            {
                "twin_tasks": len(twins),
                "memory_gap_budget": budget,
                "note": "inject a backend via fx1.eval.run_suite for live scoring",
            },
            indent=2,
        )
    )


@app.command("contamination-audit")
def contamination_audit(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/contamination_report.json")),
    with_rephrased_gap: bool = typer.Option(
        False,
        "--with-rephrased-gap",
        help="Also run the canonical-vs-rephrased gap probe against a live backend.",
    ),
    backend: str = typer.Option("hosted_k3", help="Backend for the gap probe."),
) -> None:
    """Run the publishable contamination audit over the corpus vs eval bank."""
    from fx1.eval import eval_prompt_surface, run_contamination_audit

    # Fail-closed: an absent or empty corpus certifies nothing — auditing
    # zero texts would vacuously report "not contaminated" and exit 0.
    if not corpus.exists():
        typer.echo(f"corpus not found: {corpus}; refusing to certify an empty audit", err=True)
        raise typer.Exit(code=2)
    texts = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            texts.append(" ".join(m.get("content", "") for m in record.get("messages", [])))
    if not texts:
        typer.echo(f"corpus {corpus} contains no examples; audit cannot certify", err=True)
        raise typer.Exit(code=2)
    prompts = eval_prompt_surface()
    report = run_contamination_audit(texts, prompts)
    if with_rephrased_gap:
        from fx1.eval import run_rephrased_gap
        from fx1.serve import get_backend

        model = get_backend(backend)
        report.probes.append(run_rephrased_gap(model.complete))
        report.overall_flagged = report.overall_flagged or any(p.flagged for p in report.probes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "overall_flagged": report.overall_flagged,
                "ngram_hits": len(report.ngram_hits),
                "probes": [p.method for p in report.probes],
            },
            indent=2,
        )
    )
    raise typer.Exit(code=1 if report.overall_flagged else 0)


@app.command("sign")
def sign_checkpoint(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Sign an fx-1 release (attestation ladder tier 1)."""
    from fx1.serve import sign_release

    sig = sign_release(checkpoint_dir)
    typer.echo(f"signed: {sig}")


@app.command("attestation")
def attestation_status(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Report which attestation tiers a checkpoint satisfies."""
    from fx1.serve import attestation_ladder_status

    typer.echo(json.dumps(attestation_ladder_status(checkpoint_dir), indent=2))


@app.command("sbom")
def sbom_generate(
    lockfile: Path = typer.Option(Path("uv.lock")),
    out: Path = typer.Option(Path("data/fx1/sbom.json")),
) -> None:
    """Generate a hash-pinned SBOM from the locked dependency set."""
    from fx1.sbom import generate_sbom

    sbom = generate_sbom(lockfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sbom.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"entries": len(sbom.entries), "lockfile_sha256": sbom.lockfile_sha256[:16] + "…"},
            indent=2,
        )
    )


@app.command("mrm")
def mrm_dossier(
    modelcard: Path = typer.Option(..., help="Model card JSON."),
    validation_artifact: Path = typer.Option(..., help="Contamination report JSON."),
    artifact: list[str] | None = typer.Option(
        None,
        "--artifact",
        help="Extra activity evidence as activity=path (repeatable; "
        "development/implementation/monitoring/governance).",
    ),
    out: Path = typer.Option(Path("data/fx1/mrm_dossier.json")),
) -> None:
    """Compile the five-activity model-risk dossier (SR 26-2 era)."""
    from fx1.mrm import compile_dossier

    artifacts: dict[str, str | Path] = {"validation": validation_artifact}
    for spec in artifact or []:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise typer.BadParameter("--artifact expects activity=path")
        artifacts[name] = path

    dossier = compile_dossier(
        modelcard_path=modelcard,
        artifacts=artifacts,
        out_path=out,
    )
    typer.echo(
        json.dumps(
            {
                "version": dossier.model_version,
                "complete": dossier.complete,
                "ship_eligible": dossier.ship_eligible,
                "contamination_flagged": dossier.contamination_flagged,
            },
            indent=2,
        )
    )


@app.command("dipbench")
def dipbench_demo(
    data_dir: Path | None = typer.Option(
        None, help="Directory of *_1d.parquet series for a real-data run."
    ),
    out: Path | None = typer.Option(None, help="Receipt JSON path (real-data run)."),
    threshold: float = typer.Option(0.10, help="Drawdown depth that defines a dip."),
) -> None:
    """Dip Quality Score bench. Without --data-dir: smoke the bench on a
    built-in synthetic series (SYNTHETIC — correctness only, not market
    evidence). With --data-dir: full receipt over real historical bars."""
    if data_dir is not None:
        from fx1.bench.run import run_dip_bench

        receipt = run_dip_bench(data_dir, threshold=threshold)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(
            json.dumps(
                {
                    "label": "research/backtest evidence, not live performance",
                    "events": receipt["n_events"],
                    "baseline_recovery": receipt["baseline_recovery"],
                    "receipt": str(out) if out else None,
                },
                indent=2,
            )
        )
        return
    from fx1.bench.dip import (
        DipForecast,
        detect_dip_events,
        evaluate_forecasts,
        unconditional_baseline,
    )

    closes = [100.0, 102.0, 88.0, 90.0, 103.0, 104.0, 92.0, 95.0, 106.0]
    dates = [f"2026-01-{i + 1:02d}" for i in range(len(closes))]
    events = detect_dip_events(closes, dates, "SYNTHETIC", threshold=0.10, horizons_bars={"1m": 3})
    baseline = unconditional_baseline(events)
    forecasts = [DipForecast(e.asset, e.trough_date, {"1m": 0.8}) for e in events]
    metrics = evaluate_forecasts(events, forecasts)
    typer.echo(
        json.dumps(
            {"label": "SYNTHETIC", "events": len(events), "baseline": baseline, "metrics": metrics},
            indent=2,
        )
    )


@sources_app.command("list")
def sources_list() -> None:
    """List every registered datasource with its live availability probe."""
    from fx1.data.sources.registry import list_sources
    from fx1.data.sources.router import probe_names

    specs = list_sources()
    probes = probe_names([spec.name for spec in specs])
    for spec, probe in zip(specs, probes, strict=True):
        typer.echo(
            f"{spec.name:<16} [{probe.status.value:<17}] "
            f"{spec.display} — {','.join(spec.markets)} / "
            f"{','.join(spec.assets)}"
        )


@sources_app.command("probe")
def sources_probe(
    name: str | None = typer.Argument(None, help="Source name (default: all)."),
) -> None:
    """Probe availability (script + credentials) without leaking secrets."""
    from fx1.data.sources.registry import get_spec, list_sources, roots_status
    from fx1.data.sources.router import probe_names

    specs = [get_spec(name)] if name else list_sources()
    probes = probe_names([spec.name for spec in specs])
    report = {
        "roots": roots_status(),
        "probes": [probe.model_dump() for probe in probes],
    }
    typer.echo(json.dumps(report, indent=2))


@sources_app.command("describe")
def sources_describe(name: str = typer.Argument(...)) -> None:
    """Print the source's own capability docs (straight from its CLI)."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec(name)).describe()
    if result.ok:
        typer.echo(result.text)
    else:
        typer.echo(f"unavailable: {result.error}", err=True)
        raise typer.Exit(code=1)


@sources_app.command("fetch")
def sources_fetch(
    name: str = typer.Argument(..., help="Source name."),
    api: str = typer.Option(..., help="Source-native API/tool name."),
    params_json: str = typer.Option("{}", help="JSON object of API params."),
    as_of: str | None = typer.Option(None, help="Observation date YYYY-MM-DD."),
    timeout: float = typer.Option(120.0, help="Timeout seconds (≤600)."),
) -> None:
    """Fetch from one datasource. Failure is reported, never patched over."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.base import FetchRequest
    from fx1.data.sources.registry import get_spec

    params = json.loads(params_json)
    if not isinstance(params, dict):
        typer.echo("--params-json must be a JSON object", err=True)
        raise typer.Exit(code=2)
    result = build_adapter(get_spec(name)).fetch(
        FetchRequest(api=api, params=params, as_of=as_of, timeout_s=timeout)
    )
    if result.ok:
        typer.echo(result.text)
        typer.echo(
            json.dumps(
                {
                    "payload_sha256": result.payload_sha256,
                    "fetched_at": result.fetched_at,
                    "elapsed_ms": result.elapsed_ms,
                },
                indent=2,
            ),
            err=True,
        )
    else:
        typer.echo(f"fetch failed: {result.error}", err=True)
        raise typer.Exit(code=1)


@sources_app.command("route")
def sources_route(
    question: str = typer.Option(..., help="Natural-language research question."),
    market: str | None = typer.Option(None, help="cn | hk | us | global | crypto"),
    need: str | None = typer.Option(None, help="Override need classification."),
) -> None:
    """Show the routing plan for a question (classified need + probed candidates)."""
    from fx1.data.sources.router import route

    plan = route(question, market=market, need=need)
    typer.echo(plan.model_dump_json(indent=2))


@sources_app.command("scenarios")
def sources_scenarios() -> None:
    """List finance-fetch scenario coverage (statements, consensus, peers…)."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec("finance_fetch")).describe()
    if result.ok:
        typer.echo(result.text)
    else:
        typer.echo(f"unavailable: {result.error}", err=True)
        raise typer.Exit(code=1)


@corpus_app.command("ingest-source")
def corpus_ingest_source(
    name: str = typer.Argument(..., help="Source name."),
    api: str = typer.Option(..., help="Source-native API/tool name."),
    params_json: str = typer.Option("{}"),
    as_of: str | None = typer.Option(None, help="Observation date (gated)."),
    ledger_path: Path = typer.Option(Path("data/fx1/corpus_ledger.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
) -> None:
    """Fetch → gate → append to corpus → chain into the ledger.

    Refuses (exit 1) when the fetch fails or the payload lacks an as_of date
    for time-stamped sources — no leakage, no fabrication.
    """
    from fx1.data import CorpusLedger
    from fx1.data.corpus import _system_prompt
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.base import FetchRequest
    from fx1.data.sources.ingest import fetch_to_example, record_fetch_in_ledger
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec(name)).fetch(
        FetchRequest(api=api, params=json.loads(params_json), as_of=as_of)
    )
    decision = fetch_to_example(result, _system_prompt())
    ledger = CorpusLedger(ledger_path)
    record_fetch_in_ledger(ledger, decision)
    if not decision.accepted or decision.example is None:
        typer.echo(f"ingest refused: {decision.reason}", err=True)
        raise typer.Exit(code=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(decision.example.model_dump_json() + "\n")
    typer.echo(
        json.dumps(
            {
                "accepted": True,
                "negative": decision.negative,
                "reason": decision.reason,
                "payload_sha256": decision.payload_sha256,
                "ledger": ledger.audit_export(),
            },
            indent=2,
        )
    )


@app.command("infer")
def infer_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
) -> None:
    """Run batch or walk-forward inference and write a forecast parquet.

    The fx-1 forecaster is external: ``model.name=fx-1`` requires
    ``model.entrypoint``. Reference names ``dummy-zero`` and ``dummy-momentum``
    are not fx-1. This command does not train and does not place orders.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_inference

    result = run_inference(load_harness_config(config))
    typer.echo(
        json.dumps(
            {
                "forecasts": str(result.parquet_path),
                "metadata": str(result.meta_path),
                "n_rows": result.n_rows,
                "model_name": result.metadata["model_name"],
                "model_role": result.metadata["model_role"],
                "data_label": result.metadata["data_label"],
                "research_only": True,
                "live_pnl_claim": False,
            },
            indent=2,
        )
    )


@app.command("backtest")
def backtest_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
    forecasts: Path | None = typer.Option(
        None,
        "--forecasts",
        help="Forecast parquet. Defaults to inference.output_parquet in the config.",
    ),
) -> None:
    """Score forecasts and a placeholder signal map.

    Reports forecast scores (IC, rank IC, hit rate, MAE, RMSE) and research
    diagnostics of the placeholder mapping. Does not place orders. Run
    ``fx1 infer`` first when the forecast parquet is not already on disk.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_signal_evaluation

    cfg = load_harness_config(config)
    frame = None
    if forecasts is not None:
        import polars as pl

        frame = pl.read_parquet(forecasts)
    report = run_signal_evaluation(cfg, forecasts=frame)
    typer.echo(json.dumps(report, indent=2))


@app.command("doctor")
def doctor(root: Path = typer.Option(Path("."), help="Repo root to inspect.")) -> None:
    """fx-1 readiness status (presence flags only — never secret values)."""
    from fx1.doctor import collect_status

    typer.echo(json.dumps(collect_status(root), indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
