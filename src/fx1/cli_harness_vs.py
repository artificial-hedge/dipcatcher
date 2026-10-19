"""``fx1 harness`` vector-store commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    pass

from fx1.cli_common import (
    _API_KEY_HELP,
    _FILE_ID_HELP,
    _METADATA_PAIRS_HELP,
    _ORDER_HELP,
    _REMOTE_HELP,
    _TIMEOUT_HELP,
    _VS_ID_HELP,
    _VSFB_ID_HELP,
    _bad_arg,
    _json_meta,
    _json_obj_opt,
    _or_exit,
    _remote_client,
    harness_app,
)


@harness_app.command("vs-create")
def harness_vs_create(
    name: str | None = typer.Option(None, "--name", help="Store name (free text)."),
    file_ids: str | None = typer.Option(
        None, "--file-ids", help="JSON array of file-* ids to attach at create."
    ),
    metadata: str | None = typer.Option(None, "--metadata", help=_METADATA_PAIRS_HELP),
    expires_after: str | None = typer.Option(
        None,
        "--expires-after",
        help='JSON expiry policy — e.g. {"anchor":"last_active_at","days":7}',
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores`` — mint a ``vs_*`` retrieval store the
    ``file_search`` tool searches on ``respond --tools``."""
    ids: list[str] | None = None
    if file_ids is not None:
        try:
            parsed_ids = json.loads(file_ids)
        except json.JSONDecodeError:
            _bad_arg("--file-ids must be a JSON array of file-* ids")
            raise AssertionError("unreachable") from None
        if not (isinstance(parsed_ids, list) and all(isinstance(i, str) for i in parsed_ids)):
            _bad_arg("--file-ids must be a JSON array of file-* ids")
        ids = parsed_ids
    meta = _json_meta(metadata)
    expiry = _json_obj_opt(expires_after, "--expires-after")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_create(
                name=name, metadata=meta, file_ids=ids, expires_after=expiry
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_create(
            name=name, metadata=meta, file_ids=ids, expires_after=expiry
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-get")
def harness_vs_get(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}`` — the store object; missing ids
    exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_get(vector_store_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_get(vector_store_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-update")
def harness_vs_update(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    name: str | None = typer.Option(None, "--name", help="New name (omitted keeps current)."),
    metadata: str | None = typer.Option(
        None, "--metadata", help="JSON object of string pairs — replaces wholesale."
    ),
    expires_after: str | None = typer.Option(
        None,
        "--expires-after",
        help='JSON expiry policy — e.g. {"anchor":"last_active_at","days":7}',
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}`` — set name/metadata (omitted
    fields keep their current values); ``--expires-after`` re-anchors
    the expiry window from the store's last activity."""
    meta = _json_meta(metadata)
    expiry = _json_obj_opt(expires_after, "--expires-after")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_update(
                vector_store_id, name=name, metadata=meta, expires_after=expiry
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_update(
            vector_store_id, name=name, metadata=meta, expires_after=expiry
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-delete")
def harness_vs_delete(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/vector_stores/{id}`` — drop the store and its index;
    member file-* records survive."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_delete(vector_store_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_delete(vector_store_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-list")
def harness_vs_list(
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a vs_* id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a vs_* id."),
    order: str = typer.Option("desc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores`` — stores, cursor-paged."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_list(
                limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_list(limit=limit, order=order, after=after, before=before)
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-add")
def harness_vs_file_add(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    attributes: str | None = typer.Option(
        None, "--attributes", help="JSON object — filter keys for file_search."
    ),
    chunking_strategy: str | None = typer.Option(
        None,
        "--chunking-strategy",
        help='JSON: {"type":"auto"} or {"type":"static","static":{...}}.',
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}/files`` — index a file record into
    the store; empty text lands status=failed."""
    attrs = _json_obj_opt(attributes, "--attributes")
    strat = _json_obj_opt(chunking_strategy, "--chunking-strategy")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_create(
                vector_store_id, file_id, attributes=attrs, chunking_strategy=strat
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_create(
            vector_store_id, file_id, attributes=attrs, chunking_strategy=strat
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-files")
def harness_vs_files(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a file id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a file id."),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    filter: str | None = typer.Option(
        None, "--filter", help="in_progress | completed | cancelled | failed"
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/files`` — attachments, paged;
    ``--filter`` takes an OpenAI status word."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_list(
                vector_store_id,
                limit=limit,
                after=after,
                before=before,
                order=order,
                filter=filter,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_list(
            vector_store_id, limit=limit, order=order, after=after, before=before, filter=filter
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-get")
def harness_vs_file_get(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/files/{file_id}`` — one attachment's
    status/chunks/attributes."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_get(
                vector_store_id, file_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_get(vector_store_id, file_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-delete")
def harness_vs_file_delete(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/vector_stores/{id}/files/{file_id}`` — detach; the
    file record survives."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_delete(
                vector_store_id, file_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_delete(vector_store_id, file_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-content")
def harness_vs_file_content(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/files/{file_id}/content`` — the
    stored decoded text as a page of text parts."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_content(
                vector_store_id, file_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_content(vector_store_id, file_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-search")
def harness_vs_search(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    query: str = typer.Option(..., "--query", "-q", help="Search query text."),
    max_num_results: int | None = typer.Option(
        None, "--max-results", help="Cap on returned hits (≤50)."
    ),
    filters_json: str | None = typer.Option(
        None, "--filters", help="JSON attribute-filter object."
    ),
    score_threshold: float | None = typer.Option(
        None, "--score-threshold", help="Cosine floor in [0,1]."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}/search`` — ranked hits without
    spending a response turn."""
    filters = _json_obj_opt(filters_json, "--filters")
    ranking_options = {"score_threshold": score_threshold} if score_threshold is not None else None
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_search(
                vector_store_id,
                query,
                max_num_results=max_num_results,
                filters=filters,
                ranking_options=ranking_options,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_search(
            vector_store_id,
            query,
            max_num_results=max_num_results,
            filters=filters,
            ranking_options=ranking_options,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-create")
def harness_vs_batch_create(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_ids: list[str] = typer.Argument(..., help="file-* ids to attach (1..500)."),
    attributes: str | None = typer.Option(
        None, "--attributes", help="JSON object applied to every member."
    ),
    chunking_strategy: str | None = typer.Option(
        None, "--chunking-strategy", help="JSON object (auto|static)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}/file_batches`` — attach many files
    in one call; per-file refusals count ``failed``, never abort."""
    attrs = _json_obj_opt(attributes, "--attributes")
    strategy = _json_obj_opt(chunking_strategy, "--chunking-strategy")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_create(
                vector_store_id,
                file_ids,
                attributes=attrs,
                chunking_strategy=strategy,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_batch_create(
            vector_store_id,
            file_ids,
            attributes=attrs,
            chunking_strategy=strategy,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-get")
def harness_vs_batch_get(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    batch_id: str = typer.Argument(..., help=_VSFB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/file_batches/{batch_id}`` — status +
    file_counts."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_get(
                vector_store_id, batch_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_batch_get(vector_store_id, batch_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-cancel")
def harness_vs_batch_cancel(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    batch_id: str = typer.Argument(..., help=_VSFB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST .../file_batches/{batch_id}/cancel`` — members attach
    synchronously at create, so a batch is always terminal: this
    answers 409 ``file_batch_terminal`` (exit 2)."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_cancel(
                vector_store_id, batch_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_batch_cancel(vector_store_id, batch_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-files")
def harness_vs_batch_files(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    batch_id: str = typer.Argument(..., help=_VSFB_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a file-* id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a file-* id."),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    filter: str | None = typer.Option(
        None, "--filter", help="in_progress | completed | cancelled | failed"
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET .../file_batches/{batch_id}/files`` — the frozen per-file
    verdicts in request order."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_files(
                vector_store_id,
                batch_id,
                limit=limit,
                after=after,
                before=before,
                order=order,
                filter=filter,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_batch_files(
            vector_store_id,
            batch_id,
            limit=limit,
            after=after,
            before=before,
            order=order,
            filter=filter,
        )
    )
    typer.echo(json.dumps(out, indent=2))
