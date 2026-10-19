"""``fx1 harness`` fine-tune, file, batch, and model commands."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    pass

from fx1.cli_common import (
    _API_KEY_HELP,
    _BACKEND_HELP,
    _BATCH_ID_HELP,
    _BYOK_KEY_HELP,
    _BYOK_MODEL_HELP,
    _BYOK_URL_HELP,
    _CALLBACK_SECRET_HELP,
    _FTJOB_ID_HELP,
    _LIMIT_HELP,
    _POLL_HELP,
    _REMOTE_HELP,
    _TIMEOUT_HELP,
    _bad_arg,
    _byok_opts,
    _fx1_headers,
    _need_remote,
    _or_exit,
    _remote_client,
    harness_app,
)


@harness_app.command("ft-create")
def harness_ft_create(
    training_file: str = typer.Argument(
        ...,
        help="Chat-format .jsonl corpus ({messages:[...]} per line), or a "
        "file-* id already uploaded with purpose=fine-tune (remote only).",
    ),
    model: str = typer.Option("fx1", "--model", help="Trainable model: fx1|local_fx1."),
    suffix: str | None = typer.Option(
        None, "--suffix", help="Fine-tuned model name suffix (a-z0-9_-)."
    ),
    validation_file: str | None = typer.Option(
        None, "--validation-file", help="Optional held-out .jsonl corpus, or a file-* id."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Pipeline seed."),
    epochs: int | None = typer.Option(None, "--epochs", help="n_epochs hyperparameter (1-50)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Remote only: submit and return immediately."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the job record once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
) -> None:
    """Create a gated fine-tuning job on the /v1/fine_tuning/jobs surface:
    in-process by default (SDK twin — synchronous, returns the terminal
    job), or ``--remote`` uploads the corpus (purpose=fine-tune) then
    submits and waits. Corpus validation is synchronous either way — a
    malformed file fails before the job exists."""

    def _resolve_input(raw: str, remote_mode: bool) -> tuple[str | None, bytes | None]:
        """A ``file-*`` string is an already-uploaded id (remote leg only —
        the in-process twin needs the corpus bytes); otherwise a path."""
        if raw.startswith("file-"):
            if not remote_mode:
                _bad_arg(f"{raw} is an uploaded id — the in-process leg needs a path")
            return raw, None
        p = Path(raw)
        content = p.read_bytes() if p.exists() else None
        if content is None:
            _bad_arg(f"{raw} does not exist")
        return None, content

    train_id, corpus = _resolve_input(training_file, remote is not None)
    val_id_opt: str | None = None
    val_bytes: bytes | None = None
    if validation_file is not None:
        val_id_opt, val_bytes = _resolve_input(validation_file, remote is not None)
    hp = {"n_epochs": epochs} if epochs is not None else None
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        up_id = (
            train_id
            or _or_exit(
                lambda: client.upload_file(
                    corpus or b"", filename=Path(training_file).name, purpose="fine-tune"
                )
            )["id"]
        )
        val_id = val_id_opt
        if validation_file is not None and val_id is None:
            val_id = _or_exit(
                lambda: client.upload_file(
                    val_bytes or b"",
                    filename=Path(validation_file).name,
                    purpose="fine-tune",
                )
            )["id"]
        job = _or_exit(
            lambda: client.create_finetune_job(
                model=model,
                training_file=up_id,
                hyperparameters=hp,
                suffix=suffix,
                validation_file=val_id,
                seed=seed,
                callback_url=callback_url,
                callback_secret=callback_secret,
            )
        )
        if no_wait:
            typer.echo(json.dumps(job, indent=2))
            return
        rec = _or_exit(lambda: client.wait_finetune_job(job["id"], timeout_s=None))
        typer.echo(json.dumps(rec, indent=2))
        return
    from fx1.sdk import Fx1Harness

    harness = Fx1Harness()
    from fx1.serve.finetune import FTHyperparameters

    assert corpus is not None  # _resolve_input returns bytes or exits in-process
    ftjob = _or_exit(
        lambda: harness.create_finetune_job(
            model=model,
            training_jsonl=corpus,
            validation_jsonl=val_bytes,
            hyperparameters=FTHyperparameters(**hp) if hp is not None else None,
            suffix=suffix,
            seed=seed,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    typer.echo(ftjob.model_dump_json(indent=2))


@harness_app.command("ft-jobs")
def harness_ft_jobs(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    after: str | None = typer.Option(None, "--after", help="Pagination cursor (job id)."),
) -> None:
    """List the remote fine-tuning jobs, newest first (in-process runs
    are synchronous — they return their terminal record at once)."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).finetune_jobs(
            limit=limit, after=after
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("ft-status")
def harness_ft_status(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a remote fine-tuning job's live record."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).finetune_job(job_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-events")
def harness_ft_events(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
) -> None:
    """Print a remote fine-tuning job's event feed (oldest first)."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).finetune_job_events(
            job_id, limit=limit
        )
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-checkpoints")
def harness_ft_checkpoints(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(10, "--limit", help=_LIMIT_HELP),
    after: str | None = typer.Option(None, "--after", help="Pagination cursor (ftckpt- id)."),
) -> None:
    """``GET /v1/fine_tuning/jobs/{id}/checkpoints`` — the model artifacts
    the job registered, oldest first; empty for a job that produced none."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).finetune_job_checkpoints(
            job_id, limit=limit, after=after
        )
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-wait")
def harness_ft_wait(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (job keeps running)."
    ),
) -> None:
    """Re-attach to a job submitted ``--no-wait`` and poll to terminal;
    prints the terminal record (``result_files`` carries the artifacts)."""
    _need_remote(remote)
    rec = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_finetune_job(
            job_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(json.dumps(rec, indent=2))


@harness_app.command("ft-cancel")
def harness_ft_cancel(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a remote fine-tuning job — queued cancels at once; running
    stops cooperatively at the next pipeline-stage boundary."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).cancel_finetune_job(job_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-pause")
def harness_ft_pause(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Pause a remote fine-tuning job — queued parks before starting;
    running parks at the next stage boundary. Idempotent on paused."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).pause_finetune_job(job_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-resume")
def harness_ft_resume(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Resume a paused remote fine-tuning job — restores the status
    pause captured (queued re-queues, running proceeds at its boundary)."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).resume_finetune_job(job_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("files")
def harness_files(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List the remote /v1/files store, newest first."""
    _need_remote(remote)
    typer.echo(
        json.dumps(
            _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).files()), indent=2
        )
    )


@harness_app.command("file-upload")
def harness_file_upload(
    file: Path = typer.Argument(..., help="Local .jsonl to upload."),
    purpose: str = typer.Option("batch", "--purpose", help="File purpose: batch|fine-tune."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Upload a local JSONL to /v1/files — prints the file object."""
    _need_remote(remote)
    body = file.read_bytes() if file.exists() else None
    if body is None:
        typer.echo(f"error: {file} does not exist", err=True)
        raise typer.Exit(code=2)
    up = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).upload_file(
            body, filename=file.name, purpose=purpose
        )
    )
    typer.echo(json.dumps(up, indent=2))


@harness_app.command("file-content")
def harness_file_content(
    file_id: str = typer.Argument(..., help="file- id."),
    out: Path | None = typer.Option(
        None, "--out", help="Write bytes to this path instead of stdout."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch a remote file's raw bytes (JSONL in, JSONL out)."""
    _need_remote(remote)
    blob = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).file_content(file_id))
    if out is not None:
        out.write_bytes(blob)
        typer.echo(json.dumps({"file_id": file_id, "bytes": len(blob), "out": str(out)}))
        return
    typer.echo(blob.decode("utf-8", "replace"), nl=False)


@harness_app.command("file-delete")
def harness_file_delete(
    file_id: str = typer.Argument(..., help="file- id."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Delete a remote file."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).delete_file(file_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("upload")
def harness_upload(
    file: Path = typer.Argument(..., help="Local .jsonl to ship in chunks."),
    purpose: str = typer.Option("batch", "--purpose", help="File purpose: batch|fine-tune."),
    chunk_bytes: int = typer.Option(
        4 * 1024 * 1024, "--chunk-bytes", help="Part size for /v1/uploads parts."
    ),
    md5: bool = typer.Option(
        True, "--md5/--no-md5", help="Send the content md5 on complete (fail-closed check)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Chunked upload via /v1/uploads — for files too big for one request.

    Opens the upload intent, posts the file in --chunk-bytes parts, and
    completes it into a file- record (printed on stdout; its id feeds
    batch-submit/ft-create like any /v1/files id).
    """
    _need_remote(remote)
    if chunk_bytes <= 0:
        typer.echo("error: --chunk-bytes must be positive", err=True)
        raise typer.Exit(code=2)
    body = file.read_bytes() if file.exists() else None
    if body is None:
        typer.echo(f"error: {file} does not exist", err=True)
        raise typer.Exit(code=2)
    client = _remote_client(remote or "", api_key, timeout_s)
    up = _or_exit(
        lambda: client.upload_create(
            purpose=purpose,
            filename=file.name,
            bytes=len(body),
            mime_type="application/jsonl",
        )
    )
    uid = str(up["id"])
    part_ids: list[str] = []
    for off in range(0, len(body), chunk_bytes):
        chunk = body[off : off + chunk_bytes]
        part = _or_exit(partial(client.upload_part, uid, chunk))
        part_ids.append(str(part["id"]))
    digest = hashlib.md5(body, usedforsecurity=False).hexdigest() if md5 else None
    done = _or_exit(lambda: client.upload_complete(uid, part_ids, md5=digest))
    typer.echo(json.dumps(done, indent=2))


@harness_app.command("upload-cancel")
def harness_upload_cancel(
    upload_id: str = typer.Argument(..., help="upload_ id."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a pending chunked upload (replays 200 when already cancelled)."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).upload_cancel(upload_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("batch-submit")
def harness_batch_submit(
    input_file: Path = typer.Argument(
        ..., help="Local batch-input .jsonl ({custom_id,method,url,body} per line)."
    ),
    endpoint: str = typer.Option(
        "/v1/chat/completions",
        "--endpoint",
        help="/v1/chat/completions | /v1/responses | /v1/embeddings.",
    ),
    metadata: str | None = typer.Option(
        None, "--metadata", help="JSON object of str->str batch metadata."
    ),
    idem_key: str | None = typer.Option(
        None, "--idem-key", help="Idempotency-Key — replays the submit envelope."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the batch once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Submit and return immediately (don't poll)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Upload + submit a batch on /v1/batches, then poll to terminal.
    Prints the terminal batch object (or the submitted one with
    --no-wait). The submitter's X-Fx1-* env routes every line."""
    _need_remote(remote)
    body = input_file.read_bytes() if input_file.exists() else None
    if body is None:
        typer.echo(f"error: {input_file} does not exist", err=True)
        raise typer.Exit(code=2)
    meta = None
    if metadata is not None:
        try:
            meta = json.loads(metadata)
        except ValueError as exc:
            typer.echo(f"error: --metadata is not JSON: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if not isinstance(meta, dict):
            typer.echo("error: --metadata must be a JSON object", err=True)
            raise typer.Exit(code=2)
    client = _remote_client(remote or "", api_key, timeout_s)
    up = _or_exit(lambda: client.upload_file(body, filename=input_file.name, purpose="batch"))
    sub = _or_exit(
        lambda: client.create_batch(
            up["id"],
            endpoint=endpoint,
            metadata=meta,
            idempotency_key=idem_key,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    if no_wait:
        typer.echo(json.dumps(sub, indent=2))
        return
    fin = _or_exit(lambda: client.wait_batch(sub["id"]))
    typer.echo(json.dumps(fin, indent=2))


@harness_app.command("batches")
def harness_batches(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    after: str | None = typer.Option(None, "--after", help="Pagination cursor (batch id)."),
) -> None:
    """List remote batches, newest first."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).batches(limit=limit, after=after)
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("batch-status")
def harness_batch_status(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a remote batch's live record (status + request_counts)."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).batch(batch_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("batch-wait")
def harness_batch_wait(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (batch keeps running)."
    ),
) -> None:
    """Re-attach to a batch submitted ``--no-wait`` and poll to terminal;
    prints the terminal record (``output_file_id`` feeds batch-output)."""
    _need_remote(remote)
    rec = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_batch(
            batch_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(json.dumps(rec, indent=2))


@harness_app.command("batch-cancel")
def harness_batch_cancel(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cooperative cancel — the worker checks between lines; the batch
    lands 'cancelled' with partial output in output_file_id."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).cancel_batch(batch_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("batch-output")
def harness_batch_output(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    out: Path | None = typer.Option(
        None, "--out", help="Write output JSONL bytes to this path instead of stdout."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch a finished batch's output file (output_file_id → bytes)."""
    _need_remote(remote)
    client = _remote_client(remote or "", api_key, timeout_s)
    rec = _or_exit(lambda: client.batch(batch_id))
    out_fid = rec.get("output_file_id")
    if out_fid is None:
        typer.echo(
            f"error: batch {batch_id} has no output_file_id (status {rec.get('status')})",
            err=True,
        )
        raise typer.Exit(code=2)
    blob = _or_exit(lambda: client.file_content(out_fid))
    if out is not None:
        out.write_bytes(blob)
        typer.echo(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "output_file_id": out_fid,
                    "bytes": len(blob),
                    "out": str(out),
                }
            )
        )
        return
    typer.echo(blob.decode("utf-8", "replace"), nl=False)


@harness_app.command("batch-run")
def harness_batch_run(
    input_file: Path = typer.Argument(
        ..., help="Local batch-input .jsonl ({custom_id,method,url,body} per line)."
    ),
    endpoint: str = typer.Option(
        "/v1/chat/completions",
        "--endpoint",
        help="/v1/chat/completions | /v1/responses | /v1/embeddings.",
    ),
    backend: str | None = typer.Option(
        None,
        help=_BACKEND_HELP
        + " — unset lets each line's model field resolve the link (wire parity).",
    ),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option(
        [], "--fallback", help="Alternate backend on availability faults (repeatable, max 2)."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the batch once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Write output JSONL lines to this path (default: batch object only)."
    ),
) -> None:
    """The ``batch-submit`` twin, weights-direct: run the batch input
    synchronously in-process through the same per-endpoint request
    models and gated completion core the wire worker uses — no server,
    no upload, no poll. Prints the terminal batch envelope; ``--out``
    writes the OpenAI batch-result JSONL (one line per input, ``id,
    custom_id, response:{status_code, request_id, body}, error``).
    ``--backend``/``--checkpoint-dir``/``--byok-*``/``--fallback`` map
    to the wire's X-Fx1-* headers — every line resolves the same link."""
    body = input_file.read_bytes() if input_file.exists() else None
    if body is None:
        typer.echo(f"error: {input_file} does not exist", err=True)
        raise typer.Exit(code=2)
    try:
        lines = [json.loads(ln) for ln in body.decode().splitlines() if ln.strip()]
    except ValueError as exc:
        typer.echo(f"error: {input_file} is not JSONL: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not all(isinstance(ln, dict) for ln in lines):
        typer.echo("error: every line must be a JSON object", err=True)
        raise typer.Exit(code=2)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)

    from fx1.sdk import Fx1Harness

    harness = Fx1Harness()
    batch, out_lines = _or_exit(
        lambda: harness.openai_batch(
            lines,
            endpoint=endpoint,
            headers=headers,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    if out is not None:
        out.write_text("".join(json.dumps(ln) + "\n" for ln in out_lines))
    typer.echo(json.dumps(batch, indent=2))


@harness_app.command("models")
def harness_models(
    anthropic: bool = typer.Option(
        False,
        "--anthropic",
        help="Answer in Anthropic's model-list envelope (the anthropic-version projection).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Anthropic page size (1–1000, default 20)."
    ),
    after_id: str | None = typer.Option(
        None, "--after-id", help="Anthropic cursor: page after this model id."
    ),
    before_id: str | None = typer.Option(
        None, "--before-id", help="Anthropic cursor: page before this model id."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/models`` — the OpenAI list envelope: backend names a
    ``model`` field may carry, the ``fx1`` alias, and every registered
    ``ft:`` fine-tune. ``--anthropic`` answers the anthropic-version
    projection instead (``{data, first_id, last_id, has_more}``) with
    its ``--limit``/cursor contract. In-process by default (SDK twin)."""
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if anthropic:
            out = _or_exit(
                lambda: client.anthropic_models(limit=limit, after_id=after_id, before_id=before_id)
            )
        else:
            out = _or_exit(client.list_models)
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    if anthropic:
        out = _or_exit(
            lambda: Fx1Harness().anthropic_models(
                limit=limit, after_id=after_id, before_id=before_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(Fx1Harness().openai_models().model_dump_json(indent=2))


@harness_app.command("model")
def harness_model(
    model_id: str = typer.Argument(..., help="Model id — backend name, 'fx1', or ft:name."),
    anthropic: bool = typer.Option(
        False,
        "--anthropic",
        help="Answer in Anthropic's model-card shape (the anthropic-version projection).",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/models/{id}`` — one card for a listed id; unknown ids
    fail closed (exit 2, the wire's 404). ``--anthropic`` answers the
    anthropic-version projection — ``{type: \"model\", id,
    display_name, created_at}`` — with the wire's ``not_found_error``.
    In-process by default."""
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if anthropic:
            out = _or_exit(lambda: client.anthropic_model(model_id))
            typer.echo(json.dumps(out, indent=2))
            return
        out = _or_exit(lambda: client.retrieve_model(model_id))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    if anthropic:
        out = _or_exit(lambda: Fx1Harness().anthropic_model(model_id))
        typer.echo(json.dumps(out, indent=2))
        return
    card = _or_exit(lambda: Fx1Harness().openai_model(model_id))
    typer.echo(card.model_dump_json(indent=2))


@harness_app.command("model-delete")
def harness_model_delete(
    model_id: str = typer.Argument(..., help="Registered ft: name to unregister."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/models/{id}`` — unregister an ``ft:`` fine-tune. The
    tombstone is real (the name stops resolving everywhere); built-in
    link ids refuse (exit 2, the wire's 400) and unregistered names fail
    closed (exit 2, the wire's 404). In-process by default."""
    if remote is not None:
        out = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).delete_model(model_id))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    deleted = _or_exit(lambda: Fx1Harness().openai_delete_model(model_id))
    typer.echo(deleted.model_dump_json(indent=2))
