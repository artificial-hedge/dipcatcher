"""``fx1 harness`` async job and eval-spec/run commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    pass

from fx1.cli_common import (
    _API_KEY_HELP,
    _BACKEND_HELP,
    _BYOK_KEY_HELP,
    _BYOK_MODEL_HELP,
    _BYOK_URL_HELP,
    _EVAL_ID_CREATE_HELP,
    _EVAL_ID_HELP,
    _EVAL_ID_REMOTE_HELP,
    _EVALRUN_ID_HELP,
    _LIMIT_HELP,
    _POLL_HELP,
    _REMOTE_HELP,
    _TIMEOUT_HELP,
    _byok_opts,
    _json_list_opt,
    _json_meta,
    _json_obj_opt,
    _need_remote,
    _or_exit,
    _remote_client,
    harness_app,
)


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
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
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


@harness_app.command("eval")
def harness_eval(
    suite: str = typer.Argument(
        ...,
        help="Eval suite: capability|calibration|tooluse|retrieval|ts_reasoning|ext_bench|options_reasoning.",
    ),
    backend: str = typer.Option("local_fx1", help=_BACKEND_HELP),
    seed: int = typer.Option(0, "--seed", help="Eval seed (the banks are seeded)."),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
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
        help="Alternate backend on availability faults (repeatable, max 2).",
    ),
    judge_backend: str | None = typer.Option(
        None, "--judge-backend", help="Grader link for judge suites (capability, ext_bench)."
    ),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_eval_record.v1 doc after the run."
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Remote only: submit and return immediately."
    ),
    callback_url: str | None = typer.Option(
        None,
        "--callback-url",
        help="Remote only: http(s) URL the finished eval record is POSTed to.",
    ),
    callback_secret: str | None = typer.Option(
        None,
        "--callback-secret",
        help="Remote only: HMAC secret signing the callback delivery.",
    ),
) -> None:
    """Run a seeded eval suite against a backend — in-process by default
    (SDK twin), or ``--remote`` submits to POST /harness/evals and waits
    for the terminal record. Every eval runs under the temperature=0
    pin and lands as metered evidence on the eval/completion logs."""
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    if remote is not None:
        from fx1.serve.client import HarnessClient

        client = HarnessClient(
            remote,
            api_key=api_key or os.environ.get("FX1_API_KEY") or None,
            timeout_s=timeout_s,
        )
        sub = _or_exit(
            lambda: client.submit_eval(
                suite,
                backend=backend,
                seed=seed,
                checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
                byok=byok,
                timeout_s=backend_timeout,
                fallbacks=fallbacks or None,
                judge_backend=judge_backend,
                callback_url=callback_url,
                callback_secret=callback_secret,
            )
        )
        if no_wait:
            typer.echo(json.dumps(sub, indent=2))
            return
        rec = _or_exit(lambda: client.wait_eval(sub["eval_id"], timeout_s=None))
        typer.echo(json.dumps(rec, indent=2))
        if receipt:
            doc = _or_exit(lambda: client.eval_receipt(sub["eval_id"]))
            typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    from fx1.sdk import Fx1Harness

    if callback_url is not None or callback_secret is not None:
        typer.echo(
            "--callback-url/--callback-secret are remote-only (webhooks need the server)",
            err=True,
        )
        raise typer.Exit(code=2)
    harness = Fx1Harness()
    ev_rec = _or_exit(
        lambda: harness.run_eval(
            suite,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            byok=byok,
            timeout_s=backend_timeout,
            fallbacks=fallbacks or None,
            judge_backend=judge_backend,
            seed=seed,
        )
    )
    typer.echo(ev_rec.model_dump_json(indent=2))
    if receipt:
        doc = _or_exit(lambda: harness.eval_receipt(ev_rec.eval_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))


@harness_app.command("evals")
def harness_evals(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    status: str | None = typer.Option(
        None, "--status", help="Filter: queued|running|succeeded|failed|cancelled."
    ),
    suite: str | None = typer.Option(None, "--suite", help="Filter by suite name."),
    limit: int = typer.Option(100, "--limit", help="Page size (max 256)."),
) -> None:
    """List the remote eval inventory (newest first)."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).list_evals(
            status=status, suite=suite, limit=limit
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("eval-status")
def harness_eval_status(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_REMOTE_HELP),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_eval_record.v1 doc instead."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print an eval's live record; ``--receipt`` prints the sealed
    export (409 while the eval is non-terminal)."""
    _need_remote(remote)
    if receipt:
        doc = _or_exit(
            lambda: _remote_client(remote or "", api_key, timeout_s).eval_receipt(eval_id)
        )
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).eval_status(eval_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("eval-cancel")
def harness_eval_cancel(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_REMOTE_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a queued eval; running/terminal evals report a 409 conflict."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).cancel_eval(eval_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("eval-wait")
def harness_eval_wait(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_REMOTE_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (eval keeps running)."
    ),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_eval_record.v1 doc after the record."
    ),
) -> None:
    """Re-attach to an eval submitted ``--no-wait`` and poll to terminal;
    prints the terminal record (report carries the suite output)."""
    _need_remote(remote)
    client = _remote_client(remote or "", api_key, timeout_s)
    rec = _or_exit(lambda: client.wait_eval(eval_id, poll_s=poll_s, timeout_s=wait_timeout_s))
    typer.echo(json.dumps(rec, indent=2))
    if receipt:
        doc = _or_exit(lambda: client.eval_receipt(eval_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))


@harness_app.command("eval-diff")
def harness_eval_diff(
    base_id: str = typer.Argument(..., help="Baseline eval id (terminal record)."),
    candidate_id: str = typer.Argument(..., help="Candidate eval id (terminal record)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Promotion-gate diff over two terminal evals: per-task transitions,
    gate move, by_kind deltas, and a verdict. ``comparable`` needs the
    same suite+seed and no stamped bank mismatch; non-terminal records
    report a 409 conflict."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).diff_evals(base_id, candidate_id)
    )
    typer.echo(json.dumps(st, indent=2))


# ---- /v1/evals — the OpenAI Evals-shaped spec/run surface ------------------


@harness_app.command("eval-spec-create")
def harness_eval_spec_create(
    name: str = typer.Argument(..., help="Human label for the eval container."),
    suite: str = typer.Option(
        ...,
        "--suite",
        help="Eval suite: capability|calibration|tooluse|retrieval|ts_reasoning|ext_bench|options_reasoning.",
    ),
    seed: int = typer.Option(0, "--seed", help="Eval seed (the banks are seeded)."),
    backend: str | None = typer.Option(
        None, "--backend", help="Pin the chain head (runs may still choose another model)."
    ),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    judge_backend: str | None = typer.Option(
        None, "--judge-backend", help="Grader link for judge suites."
    ),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
    criteria_json: str | None = typer.Option(
        None, "--criteria", help='testing_criteria as a JSON list, e.g. \'[{"name": "all-pass"}]\'.'
    ),
    metadata_json: str | None = typer.Option(
        None, "--metadata", help="Spec metadata as a JSON object of string pairs."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Declare a named eval container — ``POST /v1/evals`` (remote) or the
    in-process twin. The item_schema pins suite knobs; credentials never
    live on a spec."""
    criteria = _json_list_opt(criteria_json, "--criteria")
    metadata = _json_meta(metadata_json)
    if remote is not None:
        spec = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_spec_create(
                name,
                suite=suite,
                seed=seed,
                backend=backend,
                checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
                judge_backend=judge_backend,
                timeout_s=backend_timeout,
                testing_criteria=criteria,
                metadata=metadata,
            )
        )
    else:
        from fx1.sdk import Fx1Harness

        spec = _or_exit(
            lambda: Fx1Harness().eval_spec_create(
                name,
                suite=suite,
                seed=seed,
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                judge_backend=judge_backend,
                timeout_s=backend_timeout,
                testing_criteria=criteria,
                metadata=metadata,
            )
        )
    typer.echo(json.dumps(spec, indent=2))


@harness_app.command("eval-spec-list")
def harness_eval_spec_list(
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List declared eval specs (newest first) — ``GET /v1/evals``."""
    if remote is not None:
        page = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).eval_specs(limit=limit))
    else:
        from fx1.sdk import Fx1Harness

        page = {"object": "list", "data": _or_exit(lambda: Fx1Harness().eval_specs(limit=limit))}
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("eval-spec-get")
def harness_eval_spec_get(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_CREATE_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print one eval spec — ``GET /v1/evals/{id}``."""
    if remote is not None:
        spec = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).eval_spec_get(eval_id))
    else:
        from fx1.sdk import Fx1Harness

        spec = _or_exit(lambda: Fx1Harness().eval_spec_get(eval_id))
    typer.echo(json.dumps(spec, indent=2))


@harness_app.command("eval-spec-update")
def harness_eval_spec_update(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    name: str | None = typer.Option(None, "--name", help="New display name."),
    metadata_json: str | None = typer.Option(
        None, "--metadata", help="Replacement metadata as a JSON object."
    ),
    datasource_json: str | None = typer.Option(
        None,
        "--datasource",
        help="Replacement data_source_config as a JSON object — hot-reloads "
        "the declared shape only while no run binds the spec (409 once bound).",
    ),
    criteria_json: str | None = typer.Option(
        None,
        "--criteria",
        help="Replacement testing_criteria as a JSON list — same bound-freeze rule.",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Edit a spec's name/metadata, or hot-reload its declared shape while
    it has no runs — datasource/criteria freeze once a run binds."""
    metadata = _json_meta(metadata_json)
    datasource = _json_meta(datasource_json)
    criteria = _json_list_opt(criteria_json, "--criteria")
    if remote is not None:
        spec = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_spec_update(
                eval_id,
                name=name,
                metadata=metadata,
                data_source_config=datasource,
                testing_criteria=criteria,
            )
        )
    else:
        from fx1.sdk import Fx1Harness

        spec = _or_exit(
            lambda: Fx1Harness().eval_spec_update(
                eval_id,
                name=name,
                metadata=metadata,
                data_source_config=datasource,
                testing_criteria=criteria,
            )
        )
    typer.echo(json.dumps(spec, indent=2))


@harness_app.command("eval-spec-delete")
def harness_eval_spec_delete(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Delete an eval spec — journaled tombstone; bound runs stay readable."""
    if remote is not None:
        out = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).eval_spec_delete(eval_id))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    _or_exit(lambda: Fx1Harness().eval_spec_delete(eval_id))
    typer.echo(json.dumps({"id": eval_id, "object": "eval.deleted", "deleted": True}))


@harness_app.command("eval-run")
def harness_eval_run(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_CREATE_HELP),
    model: str = typer.Option(
        ...,
        "--model",
        help="Eval target: a link name (hosted_k3|local_fx1|byok), fx1, or a registered ft: name.",
    ),
    data_source_json: str | None = typer.Option(
        None,
        "--data-source",
        help='Item-schema overrides as JSON, e.g. \'{"type":"custom","source":{"seed":1}}\'.',
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help="Remote: status poll interval, seconds."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Submit a run under an eval spec — ``POST /v1/evals/{id}/runs``.
    Remote polls to the terminal run object; in-process runs are
    synchronous (the terminal object prints directly)."""
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    data_source = _json_obj_opt(data_source_json, "--data-source")
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        run = _or_exit(
            lambda: client.eval_run_create(eval_id, model=model, data_source=data_source, byok=byok)
        )
        bare = run["id"].removeprefix("evalrun_")
        _or_exit(lambda: client.wait_eval(bare, poll_s=poll_s, timeout_s=None))
        typer.echo(json.dumps(_or_exit(lambda: client.eval_run_get(eval_id, bare)), indent=2))
        return
    from fx1.sdk import Fx1Harness

    run = _or_exit(
        lambda: Fx1Harness().eval_run_create(
            eval_id,
            model=model,
            byok=byok,
            data_source_overrides=(data_source or {}).get("source"),
        )
    )
    typer.echo(json.dumps(run, indent=2))


@harness_app.command("eval-run-list")
def harness_eval_run_list(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List a spec's runs (newest first) — ``GET /v1/evals/{id}/runs``."""
    if remote is not None:
        page = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_runs(eval_id, limit=limit)
        )
    else:
        from fx1.sdk import Fx1Harness

        page = {
            "object": "list",
            "data": _or_exit(lambda: Fx1Harness().eval_runs(eval_id, limit=limit)),
        }
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("eval-run-get")
def harness_eval_run_get(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print one run — ``GET /v1/evals/{id}/runs/{run_id}``."""
    if remote is not None:
        run = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_run_get(eval_id, run_id)
        )
    else:
        from fx1.sdk import Fx1Harness

        run = _or_exit(lambda: Fx1Harness().eval_run_get(eval_id, run_id))
    typer.echo(json.dumps(run, indent=2))


@harness_app.command("eval-run-cancel")
def harness_eval_run_cancel(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a queued run — ``POST .../runs/{run_id}/cancel`` (remote only:
    in-process runs are synchronous, already terminal)."""
    _need_remote(remote)
    run = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).eval_run_cancel(eval_id, run_id)
    )
    typer.echo(json.dumps(run, indent=2))


@harness_app.command("eval-run-delete")
def harness_eval_run_delete(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Delete a terminal run record — ``DELETE .../runs/{run_id}``."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_run_delete(eval_id, run_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    _or_exit(lambda: Fx1Harness().eval_run_delete(eval_id, run_id))
    typer.echo(json.dumps({"id": run_id, "object": "eval.run.deleted", "deleted": True}))


@harness_app.command("eval-run-items")
def harness_eval_run_items(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Per-task verdict rows from a completed run — ``.../output_items``."""
    if remote is not None:
        page = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_run_output_items(
                eval_id, run_id, limit=limit
            )
        )
    else:
        from fx1.sdk import Fx1Harness

        page = {
            "object": "list",
            "data": _or_exit(lambda: Fx1Harness().eval_run_items(eval_id, run_id, limit=limit)),
        }
    typer.echo(json.dumps(page, indent=2))
