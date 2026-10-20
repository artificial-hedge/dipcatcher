"""``fx1 harness`` core operator commands (inspect, serve, keys, health)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

if TYPE_CHECKING:
    from fx1.sdk import Fx1Harness
    from fx1.serve.client import HarnessClient

from fx1.cli_common import (
    _API_KEY_HELP,
    _BACKEND_HELP,
    _BYOK_KEY_HELP,
    _BYOK_MODEL_HELP,
    _BYOK_URL_HELP,
    _OPERATION_ARGUMENT_MAX_BYTES,
    _REMOTE_HELP,
    _TIMEOUT_HELP,
    _byok_opts,
    _last_request_id,
    _load_prompts,
    _or_exit,
    _record_json,
    _remote_client,
    _surface,
    harness_app,
)


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

    # a bogus --role is an arg fault on the same clean line the remote leg
    # gets from _or_exit — never a raw traceback.
    role_filter = _or_exit(lambda: HarnessRole(role)) if role else None
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

        # same refusal shape as the remote leg: registry misses print one
        # clean error line + exit 2, never a traceback exit 1.
        result = _or_exit(lambda: Harness().run(name))
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    raise typer.Exit(code=result.exit_code)


@harness_app.command("doctor")
def harness_doctor(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    receipts_dir: Path = typer.Option(
        Path("receipts"),
        "--receipts-dir",
        help="Local leg: sealed-receipt store for the round-trip check.",
    ),
    state_dir: Path | None = typer.Option(
        None,
        "--state-dir",
        help="Local leg: also probe durable-state replay on this dir.",
    ),
) -> None:
    """Run the harness diagnostic battery; exit 1 when any check fails.

    With ``--remote`` the battery diagnoses the deployment over the wire
    (readiness, health, wire-contract negotiation, auth enforcement, the
    managed-key lifecycle, metrics, a sealed-receipt round-trip); without
    it the same report is produced in-process against ``Fx1Harness`` —
    honesty gate, a gated completion through a deterministic injected
    backend, the managed-key lifecycle, and a sealed-receipt round-trip.
    Checks a target legitimately cannot answer report ``skipped`` and do
    not fail the run. Secrets never appear in the output.
    """
    from fx1.serve.harness_doctor import run_local_doctor, run_remote_doctor

    if remote is not None:
        from fx1.serve.client import HarnessClient

        def make_client(key: str | None) -> HarnessClient:
            return HarnessClient(remote, api_key=key, timeout_s=timeout_s)

        report = run_remote_doctor(
            make_client(api_key or os.environ.get("FX1_API_KEY") or None),
            make_client,
        )
    else:
        report = run_local_doctor(
            receipts_dir=receipts_dir,
            state_dir=state_dir,
        )
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(code=0 if report["ok"] else 1)


@harness_app.command("serve")
def harness_serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8011, help="Bind port."),
    public_release: bool = typer.Option(
        False,
        "--public-release",
        help="Enforce public model, auth, and release gates on a loopback listener behind a proxy.",
    ),
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
    store_max: int | None = typer.Option(
        None,
        help="/v1 retrieval-index capacity (env FX1_API_STORE_MAX, default 256).",
    ),
    state_dir: str | None = typer.Option(
        None,
        help="Durable state dir — journals async-job transitions across restarts "
        "(env FX1_API_STATE_DIR; unset = in-memory only).",
    ),
) -> None:
    """Serve the harness API (POST /harness/runs, /harness/complete, /receipts/verify)."""
    import uvicorn

    from fx1.serve.api import create_app
    from fx1.serve.release_gate import ALLOW_TEST_FIXTURE_ENV, cli_public_release_mode

    try:
        public_release = cli_public_release_mode(host, explicit=public_release)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get("FX1_API_KEY"):
        typer.echo(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure",
            err=True,
        )
        raise typer.Exit(code=2)
    if public_release and os.environ.get(ALLOW_TEST_FIXTURE_ENV) == "1":
        typer.echo("test fixture opt-in may bind only to loopback research mode", err=True)
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
            store_max=store_max,
            state_dir=state_dir,
            public_release=public_release,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    uvicorn.run(harness_api, host=host, port=port, reload=False, server_header=False)


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
    fallbacks: list[str] = typer.Option(
        [],
        "--fallback",
        help="Alternate backend to try on availability faults (repeatable, max 2).",
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="Decode temperature (default 0.0 — deterministic)."
    ),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling mass (0,1]."),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Completion token cap sent to the provider."
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Decode seed passed to providers that support it."
    ),
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
                fallbacks=fallbacks or None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
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
            fallbacks=fallbacks or None,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
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
    fallbacks: list[str] = typer.Option(
        [],
        "--fallback",
        help="Alternate backend to try on availability faults (repeatable, max 2).",
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="Decode temperature (default 0.0 — deterministic)."
    ),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling mass (0,1]."),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Completion token cap sent to the provider."
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Decode seed passed to providers that support it."
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
            fallbacks=fallbacks or None,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
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
        # one round-trip on the wire (receipts/verify/batch); the SDK twin
        # loops in-process — same per-item verdict shape either way.
        verdicts = _or_exit(
            lambda: surface.verify_receipts([json.loads(f.read_text()) for f in files])
        )
        results = [{"file": f.name, "valid": v.valid} for f, v in zip(files, verdicts, strict=True)]
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


@harness_app.command("score")
def harness_score(
    inputs: list[str] = typer.Argument(..., help="Text to score (repeatable)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /harness/score`` — the deterministic reward contract as a
    preflight surface: per-input gate categories + reward signal, no model
    spend. In-process by default."""
    inp: str | list[str] = inputs[0] if len(inputs) == 1 else list(inputs)
    surface = _surface(remote, api_key, timeout_s)
    out = _or_exit(lambda: surface.score(inp))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("commands")
def harness_commands(
    role: str | None = typer.Option(
        None, "--role", help="Filter to one role's reachable commands."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /harness/commands`` — registered command names; anything
    unlisted is unreachable on that role."""
    surface = _surface(remote, api_key, timeout_s)
    # lazy: keeps the fx1.harness import out of CLI startup
    from fx1.harness import HarnessRole

    role_val: HarnessRole | None = None
    if role is not None:
        # bogus role names are an arg fault, same as the wire's 422.
        role_val = _or_exit(lambda: HarnessRole(role))
    names = _or_exit(lambda: surface.commands(role=role_val))
    typer.echo(json.dumps({"count": len(names), "commands": names}, indent=2))


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
        out = _or_exit(lambda: client.server_version())
        # the wire's request-id for this very call — the trace surface a
        # deploy loop would paste into a bug report
        rid = _last_request_id(client)
        if rid is not None:
            out = {**out, "request_id": rid}
        typer.echo(json.dumps(out))
        return
    from fx1 import __version__
    from fx1.serve.contract import API_VERSION

    typer.echo(json.dumps({"api_version": API_VERSION, "fx1_version": __version__, "local": True}))


@harness_app.command("selftest")
def harness_selftest(
    remote: str | None = typer.Option(
        None,
        "--remote",
        help="Smoke a live deployment (read-only checks); unset boots a "
        "stub engine + the production app on loopback and runs the full "
        "golden path.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    state_dir: str | None = typer.Option(
        None,
        "--state-dir",
        help="Also prove durability: restart the local app on this dir and "
        "recover the job record (local mode only).",
    ),
) -> None:
    """Golden-path smoke: zero-config proof the harness actually works.

    Local mode spins a stub OpenAI engine and the production app on
    loopback, then walks auth, commands, a BYOK completion, SSE stream,
    idempotent replay, the async job lifecycle, sealed-receipt verify,
    drain, and in-process parity. Remote mode runs read-only checks only.
    Exits 0 when every check passes, 2 otherwise — usable as a deploy gate."""
    from fx1.selftest import run_selftest

    report = run_selftest(
        remote=remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        state_dir=state_dir,
        timeout_s=timeout_s,
    )
    typer.echo(json.dumps(report.as_dict(), indent=2))
    if not report.ok:
        raise typer.Exit(code=2)


@harness_app.command("bench")
def harness_bench(
    remote: str | None = typer.Option(
        None,
        "--remote",
        help="Bench a live deployment; unset times the in-process SDK.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    n: int = typer.Option(32, "--n", help="Measured requests (1..4096)."),
    concurrency: int = typer.Option(4, "--concurrency", help="Worker pool size (1..256)."),
    warmup: int = typer.Option(2, "--warmup", help="Unmeasured warmup requests (0..256)."),
    prompt: str = typer.Option(
        "", "--prompt", help="Probe prompt (default: a fixed summary prompt)."
    ),
    max_tokens: int = typer.Option(32, "--max-tokens", help="Per-request token cap."),
    backend: str = typer.Option("local_fx1", "--backend", help=_BACKEND_HELP),
    seed: int | None = typer.Option(
        None, "--seed", help="Sampling seed forwarded to each request."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    receipt: bool = typer.Option(
        False,
        "--receipt",
        help="Print the sealed fx1_bench_result.v1 doc instead of the raw record.",
    ),
) -> None:
    """Perf gate: time ``n`` gated completions at ``--concurrency`` workers.

    Prints the latency card (p50/p90/p95/p99/max/mean), throughput, token
    rates, and an error histogram as JSON; ``--receipt`` prints the sealed
    fx1_bench_result.v1 document (verify with ``dipcatcher verify-receipt``).
    Exits 0 when every measured request succeeded, 1 when any failed —
    usable as a deploy gate alongside ``harness selftest``."""
    from fx1.harness_bench import DEFAULT_BENCH_PROMPT, run_bench
    from fx1.serve.ops_receipt import bench_receipt

    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    record = _or_exit(
        lambda: run_bench(
            surface,
            n=n,
            concurrency=concurrency,
            warmup=warmup,
            prompt=prompt or DEFAULT_BENCH_PROMPT,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            backend=backend,
            byok=_byok_opts(byok_base_url, byok_api_key, byok_model),
            seed=seed,
            mode="remote" if remote else "in_process",
        )
    )
    typer.echo(json.dumps(bench_receipt(record) if receipt else record, indent=2, sort_keys=True))
    if record["metrics"]["error_count"]:
        raise typer.Exit(code=1)


@harness_app.command("usage")
def harness_usage(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_HELP),
    model: str | None = typer.Option(None, "--model", help="Only count this reported model."),
    since: float | None = typer.Option(
        None, "--since", help="Unix-second lower bound on record timestamps."
    ),
    until: float | None = typer.Option(
        None, "--until", help="Unix-second upper bound on record timestamps."
    ),
    key_id: str | None = typer.Option(
        None, "--key-id", help="Only count calls made under this key fingerprint."
    ),
) -> None:
    """Usage accounting: token/request aggregates over the completion log.

    Totals plus per-backend/per-model splits as JSON — the billing/ops
    view of every gated call the surface served. ``records_dropped`` in
    the report declares a truncated ring window. ``--remote`` reads the
    server's log (``GET /harness/usage``); default reads the in-process
    SDK's own log."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    report = _or_exit(
        lambda: surface.usage(backend=backend, model=model, key_id=key_id, since=since, until=until)
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


@harness_app.command("key-create")
def harness_key_create(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    name: str | None = typer.Option(None, "--name", help="Label for the key (≤128 chars)."),
    admin: bool = typer.Option(
        False,
        "--admin",
        help="Mint an admin key — it may itself mint/list/revoke keys.",
    ),
    rpm: int | None = typer.Option(
        None,
        "--rpm",
        min=1,
        help="Bound the key to this many requests per 60 s window (over-limit → 429).",
    ),
    ttl_s: float | None = typer.Option(
        None,
        "--ttl-s",
        help="Key expiry in seconds from mint — the credential dies after.",
    ),
    scopes: list[str] = typer.Option(
        [],
        "--scope",
        help="Bound the key to a surface class: read | write | admin (repeatable; unset = read+write).",
    ),
    max_requests: int | None = typer.Option(
        None,
        "--max-requests",
        min=1,
        help="Hard budget: the key refuses 429 quota_exceeded after this many authenticated calls.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        min=1,
        help="Hard budget: refuses once the key's provider-reported token spend reaches this.",
    ),
) -> None:
    """Mint a managed API key — prints the mint record including the raw
    ``key``, which is shown once and never stored server-side. On
    ``--remote`` this needs the bootstrap credential (FX1_API_KEY)."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(
        lambda: surface.key_create(
            name,
            admin=admin,
            rpm=rpm,
            ttl_s=ttl_s,
            scopes=scopes or None,
            max_requests=max_requests,
            max_tokens=max_tokens,
        )
    )
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("keys")
def harness_keys(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List managed API keys — fingerprint ids + metadata, never secrets."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.keys())
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-get")
def harness_key_get(
    key_id: str,
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """One managed key's record by its fingerprint id."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_get(key_id))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-revoke")
def harness_key_revoke(
    key_id: str,
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Tombstone a managed key — auth with it fails closed immediately
    after; the record stays for audit."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_revoke(key_id))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-usage")
def harness_key_usage(
    key_id: str,
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """One managed key's usage card — live counters, budget headroom,
    rate-window state, and the completion-ring spend split. Admin on
    the wire."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_usage(key_id))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-rotate")
def harness_key_rotate(
    key_id: str,
    revoke_old: bool = typer.Option(
        True,
        "--revoke-old/--keep-old",
        help="Tombstone the predecessor atomically with the mint (default). "
        "--keep-old leaves both secrets live until the old key is revoked or expires.",
    ),
    name: str | None = typer.Option(
        None, "--name", help="Successor display name — defaults to the predecessor's."
    ),
    ttl_s: float | None = typer.Option(
        None,
        "--ttl-s",
        min=1e-9,
        help="Fresh lifetime for the successor (seconds). Omitted: inherits the "
        "predecessor's absolute expires_at — rotation never extends a credential.",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Rotate a managed key — mints a successor under the predecessor's
    declared policy and, by default, tombstones the old secret in the
    same transaction. The response's ``key.key`` is the only place the
    new secret appears."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(
        lambda: surface.key_rotate(key_id, revoke_old=revoke_old, name=name, ttl_s=ttl_s)
    )
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-patch")
def harness_key_patch(
    key_id: str,
    name: str | None = typer.Option(
        None, "--name", help="New display name — unset keeps the declared name."
    ),
    rpm: int | None = typer.Option(
        None,
        "--rpm",
        min=1,
        help="New per-60s request bound — unset keeps the declared one.",
    ),
    scopes: list[str] = typer.Option(
        [],
        "--scope",
        help="Replace the surface classes: read | write | admin (repeatable).",
    ),
    admin: bool | None = typer.Option(
        None,
        "--admin/--no-admin",
        help="--admin unions the admin scope onto the surviving list; "
        "--no-admin never strips a declared scope (purely additive, like mint).",
    ),
    max_requests: int | None = typer.Option(
        None, "--max-requests", min=1, help="New authenticated-call budget."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", min=1, help="New provider-reported token budget."
    ),
    expires_at: float | None = typer.Option(
        None,
        "--expires-at",
        min=1e-9,
        help="New absolute expiry (unix seconds).",
    ),
    clear: list[str] = typer.Option(
        [],
        "--clear",
        help="Clear a nullable bound back to unbounded: name | rpm | "
        "max_requests | max_tokens | expires_at (repeatable).",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Patch a live managed key's policy in place — prints the updated
    record. Unset flags keep the declared policy; ``--clear FIELD``
    sends the JSON-null that unbounds it; ``enabled`` and the live
    counters are not patchable — revocation is permanent."""
    kw: dict[str, Any] = {}
    if name is not None:
        kw["name"] = name
    if rpm is not None:
        kw["rpm"] = rpm
    if scopes:
        kw["scopes"] = scopes
    if admin is not None:
        kw["admin"] = admin
    if max_requests is not None:
        kw["max_requests"] = max_requests
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if expires_at is not None:
        kw["expires_at"] = expires_at
    if clear:
        kw["clear"] = clear
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_update(key_id, **kw))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("self")
def harness_self(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """The calling credential's own card — its class (managed / env /
    loopback) plus, for managed keys, live budget headroom. Read scope."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.self_usage())
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


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
    rid = _last_request_id(client)
    if rid is not None:
        report = {**report, "request_id": rid}
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
