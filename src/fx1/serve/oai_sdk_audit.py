"""oai_sdk_audit — the stock ``openai`` Python SDK as the drop-in leg.

The harness advertises an OpenAI-compatible wire surface; ``HarnessClient``
is our own client and could silently drift in our favor. This audit points
the *unmodified* official ``openai.AsyncOpenAI`` at the in-process ASGI
app — no socket, one deterministic stub backend — and asserts SDK-typed
results throughout: a response that parses into the SDK's own models is
the compatibility claim; a typed ``APIStatusError`` subclass carrying the
``error`` envelope is the error-shape claim.

Pinned contract (one probe per row):

- *models* — ``list``/``retrieve`` parse; ``delete`` on a built-in link
  is a typed 400, on an unknown name a typed 404.
- *chat.completions* — ``create`` parses (message content, finish_reason,
  usage); ``stream=True`` parses chunks; ``stream_options.include_usage``
  lands the final usage chunk; stored completions
  ``retrieve``/``list``/``messages.list``/``delete`` round-trip, then 404.
- *responses* — ``create`` parses ``status``/``output``; ``stream=True``
  parses the typed event stream to ``response.completed``; ``retrieve``,
  ``input_items.list``, ``delete``, ``cancel`` round-trip; a deleted
  envelope reads ``NotFoundError``.
- *files* — multipart ``create`` (``purpose=fine-tune``/``batch``),
  ``retrieve``/``list``/``content``/``delete``; a bad purpose is 400.
- *batches* — ``create``/``retrieve``/``list`` over an uploaded ``.jsonl``.
- *fine_tuning.jobs* — ``create``/``retrieve``/``list``/``cancel``/
  ``list_events``/``checkpoints.list``; an untrainable model is 400.
- *vector_stores* — ``create`` (incl. ``expires_after`` via
  ``extra_body``)/``retrieve``/``update``/``list``/``delete``/``search``;
  ``files.*`` and ``file_batches.*`` round-trip.
- *evals* — spec ``create``/``retrieve``/``list``/``update``/``delete``
  and ``runs.create``/``list`` on the OpenAI Evals-shaped surface.
- *conversations* — ``create``/``retrieve``/``update``/``delete`` plus
  ``items.create``/``list``/``retrieve``/``delete``.
- *uploads* — ``create``/``parts.create``/``complete``/``cancel``.
- *embeddings* — float and ``base64`` bodies parse.
- *moderations* — ``results[0].flagged`` parses.
- *errors* — the shared ``{error: {message, type, code}}`` envelope maps
  to ``NotFoundError``/``BadRequestError``/``UnprocessableEntityError``.

Sealed ``oai_sdk_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fx1.serve.backends import SamplingParams

__all__ = ["oai_sdk_audit", "oai_sdk_audit_bench"]

_STUB_HI = "stub:hi"

_FT_JSONL = (
    b'{"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"}]}\n'
    b'{"messages":[{"role":"user","content":"q2"},{"role":"assistant","content":"a2"}]}\n'
)
_BATCH_JSONL = (
    b'{"custom_id":"c1","method":"POST","url":"/v1/chat/completions",'
    b'"body":{"model":"fx1","messages":[{"role":"user","content":"hi"}]}}\n'
)


class _SdkBackend:
    """Deterministic stub — the compatibility claim is about the wire
    layer, so the backend is a fixed echo with stream + embeddings
    channels (never a real model)."""

    def __init__(self) -> None:
        self._model = "stub-0"
        # the usage channel the route reads after each call — a stubbed
        # count so the response surface carries a populated usage object
        self.last_usage: dict[str, int] | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        return f"stub:{messages[-1]['content']}"

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        yield "stub:"
        yield "alpha"

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> Any:
        from fx1.serve.backends import EmbeddingResult

        n = (
            len(input)
            if isinstance(input, list) and input and isinstance(input[0], (str, list))
            else 1
        )
        if encoding_format == "base64":
            data = tuple(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": base64.b64encode(b"\x00\x00\x80?" * 8).decode(),
                }
                for i in range(n)
            )
        else:
            data = tuple(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [0.1 * (i + 1)] * 4,
                }
                for i in range(n)
            )
        return EmbeddingResult(
            data=data,
            model=model,
            usage={"prompt_tokens": 4, "total_tokens": 4},
        )


@contextmanager
def _sdk_client() -> Iterator[Any]:
    """``AsyncOpenAI`` bound to the in-process app over httpx's ASGI
    transport — no socket; env keys cleared so the loopback-authorized
    surface answers unauthenticated (the auth gates are probed in
    ``api_audit``)."""
    # httpx2 is openai's vendored httpx fork — the http_client must be a
    # httpx2.AsyncClient for the SDK's strict type + runtime checks
    import httpx2
    import openai

    from fx1.serve import api as api_mod

    saved = {k: os.environ.get(k) for k in ("FX1_API_KEY", "MOONSHOT_API_KEY")}
    try:
        os.environ.pop("FX1_API_KEY", None)
        os.environ.pop("MOONSHOT_API_KEY", None)
        app = api_mod.create_app(backend_resolver=lambda *a, **k: _SdkBackend())
        transport = httpx2.ASGITransport(app=app)
        http = httpx2.AsyncClient(transport=transport, base_url="http://sdk-audit")
        yield openai.AsyncOpenAI(
            base_url="http://sdk-audit/v1",
            api_key="sdk-audit",
            http_client=http,
            max_retries=0,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _probe_models(cl: Any, out: dict[str, Any]) -> None:
    import openai

    models = await cl.models.list()
    ids = sorted(m.id for m in models.data)
    out["sdk_models_list"] = {"fx1", "hosted_k3", "local_fx1", "byok"} <= set(ids) and all(
        m.object == "model" for m in models.data
    )
    got = await cl.models.retrieve("fx1")
    out["sdk_models_retrieve"] = got.id == "fx1" and got.object == "model"
    try:
        await cl.models.delete("fx1")
        out["sdk_models_delete_builtin_400"] = False
    except openai.BadRequestError:
        out["sdk_models_delete_builtin_400"] = True
    try:
        await cl.models.delete("definitely-not-a-model")
        out["sdk_models_delete_unknown_404"] = False
    except openai.NotFoundError:
        out["sdk_models_delete_unknown_404"] = True


async def _probe_chat(cl: Any, out: dict[str, Any]) -> None:
    import openai

    cc = await cl.chat.completions.create(model="fx1", messages=[{"role": "user", "content": "hi"}])
    out["sdk_chat_create"] = (
        cc.object == "chat.completion"
        and cc.choices[0].message.content == _STUB_HI
        and cc.choices[0].finish_reason == "stop"
        and cc.usage is not None
        and cc.usage.total_tokens > 0
    )
    chunks = [
        c
        async for c in await cl.chat.completions.create(
            model="fx1", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    ]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    out["sdk_chat_stream"] = (
        bool(chunks)
        and bool(chunks[-1].choices)
        and chunks[-1].choices[0].finish_reason == "stop"
        # the stream surface chunkifies the completed text — the SDK's
        # delta-channel compat is what the probe claims
        and text == _STUB_HI
    )
    uchunks = [
        c
        async for c in await cl.chat.completions.create(
            model="fx1",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
    ]
    out["sdk_chat_stream_usage"] = (
        bool(uchunks)
        and uchunks[-1].choices == []
        and uchunks[-1].usage is not None
        and uchunks[-1].usage.total_tokens > 0
    )
    stored = await cl.chat.completions.create(
        model="fx1", messages=[{"role": "user", "content": "store-me"}], store=True
    )
    fetched = await cl.chat.completions.retrieve(stored.id)
    out["sdk_chat_retrieve"] = (
        fetched.id == stored.id and fetched.choices[0].message.content == "stub:store-me"
    )
    listed = [c async for c in cl.chat.completions.list(limit=50)]
    out["sdk_chat_list"] = any(c.id == stored.id for c in listed)
    msgs = [m async for m in cl.chat.completions.messages.list(stored.id)]
    out["sdk_chat_messages"] = any("store-me" in str(m.content) for m in msgs)
    updated = await cl.chat.completions.update(stored.id, metadata={"t": "1"})
    refetched = await cl.chat.completions.retrieve(stored.id)
    out["sdk_chat_update"] = (
        updated.id == stored.id
        and dict(updated.metadata or {}) == {"t": "1"}
        and dict(refetched.metadata or {}) == {"t": "1"}
    )
    deleted = await cl.chat.completions.delete(stored.id)
    out["sdk_chat_delete"] = deleted.id == stored.id and deleted.deleted is True
    try:
        await cl.chat.completions.retrieve(stored.id)
        out["sdk_chat_deleted_404"] = False
    except openai.NotFoundError:
        out["sdk_chat_deleted_404"] = True


async def _probe_completions(cl: Any, out: dict[str, Any]) -> None:
    import openai

    # the legacy text surface — what `client.completions.create` and
    # pre-chat tooling drive
    cm = await cl.completions.create(model="fx1", prompt="hi")
    out["sdk_completion_create"] = (
        cm.object == "text_completion"
        and cm.id.startswith("cmpl-")
        and cm.choices[0].text == _STUB_HI
        and cm.choices[0].finish_reason == "stop"
    )
    multi = await cl.completions.create(model="fx1", prompt=["a", "b"], n=2)
    out["sdk_completion_multi_prompt"] = (
        len(multi.choices) == 4
        and [c.index for c in multi.choices] == [0, 1, 2, 3]
        and multi.choices[0].text == "stub:a"
        and multi.choices[2].text == "stub:b"
    )
    echoed = await cl.completions.create(model="fx1", prompt="hi", echo=True)
    out["sdk_completion_echo"] = echoed.choices[0].text == "hi" + _STUB_HI
    schunks = [c async for c in await cl.completions.create(model="fx1", prompt="hi", stream=True)]
    out["sdk_completion_stream"] = (
        bool(schunks)
        and all(c.object == "text_completion" for c in schunks)
        and "".join(c.choices[0].text for c in schunks if c.choices) == _STUB_HI
        and schunks[-1].choices[0].finish_reason == "stop"
    )
    # legacy-only fields refuse typed — the SDK maps our 422 to
    # UnprocessableEntityError with the openai-shaped body
    try:
        await cl.completions.create(model="fx1", prompt="x", extra_body={"suffix": "s"})
        out["sdk_completion_suffix_422"] = False
    except openai.UnprocessableEntityError as exc:
        out["sdk_completion_suffix_422"] = exc.status_code == 422


async def _probe_responses(cl: Any, out: dict[str, Any]) -> None:
    import openai

    r = await cl.responses.create(model="fx1", input="hi")
    out["sdk_resp_create"] = (
        r.object == "response"
        and r.status == "completed"
        and r.output[0].type == "message"
        and r.usage is not None
    )
    events = [ev async for ev in await cl.responses.create(model="fx1", input="hi", stream=True)]
    out["sdk_resp_stream"] = (
        bool(events)
        and events[0].type == "response.created"
        and events[-1].type == "response.completed"
        and events[-1].response.status == "completed"
    )
    fetched = await cl.responses.retrieve(r.id)
    items = [it async for it in cl.responses.input_items.list(r.id)]
    out["sdk_resp_retrieve_items"] = (
        fetched.id == r.id and bool(items) and items[0].type == "message"
    )
    queued = await cl.responses.create(model="fx1", input="hi", background=True)
    out["sdk_resp_background_queued"] = queued.status in ("queued", "in_progress", "completed")
    try:
        done = await cl.responses.cancel(queued.id)
        out["sdk_resp_cancel"] = done.id == queued.id and done.status in (
            "cancelled",
            "completed",
            "failed",
        )
    except openai.ConflictError:
        # already terminal — the cancel route still answered the typed
        # 409 envelope
        out["sdk_resp_cancel"] = True
    deleted = await cl.responses.delete(r.id)
    out["sdk_resp_delete"] = deleted is None or getattr(deleted, "id", None) == r.id
    try:
        await cl.responses.retrieve(r.id)
        out["sdk_resp_deleted_404"] = False
    except openai.NotFoundError:
        out["sdk_resp_deleted_404"] = True
    # retrieve(stream=True) — the stock SDK's replay-shaped call: the
    # typed event stream rebuilds the same Response the non-stream
    # retrieve returned (the real drop-in conformance claim), and the
    # grammar is identical to a create-time stream.
    r2 = await cl.responses.create(model="fx1", input="hi")
    replayed = [ev async for ev in await cl.responses.retrieve(r2.id, stream=True)]
    out["sdk_resp_replay"] = (
        bool(replayed)
        and replayed[0].type == "response.created"
        and replayed[-1].type == "response.completed"
        and replayed[-1].response.id == r2.id
        and replayed[-1].response.status == "completed"
        and replayed[-1].response.output[0].content[0].text == r2.output[0].content[0].text
        and replayed[-1].response.usage is not None
    )
    # starting_after=N slices to events with sequence > N — the prelude
    # is skipped and the terminal frame still parses typed.
    sliced = [ev async for ev in await cl.responses.retrieve(r2.id, stream=True, starting_after=1)]
    out["sdk_resp_replay_starting_after"] = (
        bool(sliced)
        and sliced[0].type != "response.created"
        and sliced[-1].type == "response.completed"
        and sliced[-1].response.id == r2.id
        and len(sliced) == len(replayed) - 2
    )


async def _probe_files_batches_ft(cl: Any, out: dict[str, Any]) -> None:
    import openai

    f = await cl.files.create(file=("train.jsonl", _FT_JSONL), purpose="fine-tune")
    out["sdk_file_create"] = f.object == "file" and f.id.startswith("file-")
    out["sdk_file_retrieve"] = (await cl.files.retrieve(f.id)).id == f.id
    listed = [x async for x in cl.files.list()]
    out["sdk_file_list"] = any(x.id == f.id for x in listed)
    content = await cl.files.content(f.id)
    out["sdk_file_content"] = b'"messages"' in content.content
    try:
        await cl.files.create(file=("bad.jsonl", b"x"), purpose="assistants")
        out["sdk_file_bad_purpose_400"] = False
    except openai.BadRequestError:
        out["sdk_file_bad_purpose_400"] = True

    bf = await cl.files.create(file=("batch.jsonl", _BATCH_JSONL), purpose="batch")
    b = await cl.batches.create(
        input_file_id=bf.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    out["sdk_batch_create"] = b.object == "batch" and b.id.startswith("batch_")
    got_b = await cl.batches.retrieve(b.id)
    out["sdk_batch_retrieve"] = got_b.id == b.id
    batches = [x async for x in cl.batches.list(limit=10)]
    out["sdk_batch_list"] = any(x.id == b.id for x in batches)

    job = await cl.fine_tuning.jobs.create(
        model="fx1", training_file=f.id, hyperparameters={"n_epochs": 1}
    )
    out["sdk_ft_create"] = job.object == "fine_tuning.job" and job.id.startswith("ftjob-")
    out["sdk_ft_retrieve"] = (await cl.fine_tuning.jobs.retrieve(job.id)).id == job.id
    jobs = [j async for j in cl.fine_tuning.jobs.list(limit=10)]
    out["sdk_ft_list"] = any(j.id == job.id for j in jobs)
    events = [e async for e in cl.fine_tuning.jobs.list_events(job.id, limit=10)]
    out["sdk_ft_events"] = all(e.object == "fine_tuning.job.event" for e in events)
    ckpts = [c async for c in cl.fine_tuning.jobs.checkpoints.list(job.id)]
    out["sdk_ft_checkpoints"] = isinstance(ckpts, list)
    # cooperative cancel: a queued job flips now, a running one is marked
    # and lands terminal at the next stage boundary — and a fast job may
    # already be terminal (the API then answers a typed 409 Conflict)
    try:
        cancelled = await cl.fine_tuning.jobs.cancel(job.id)
        cancel_seen = cancelled.id == job.id
    except openai.ConflictError:
        cancel_seen = True
    terminal = ""
    for _ in range(60):
        got = await cl.fine_tuning.jobs.retrieve(job.id)
        if got.status in ("cancelled", "succeeded", "failed"):
            terminal = got.status
            break
        await asyncio.sleep(0.5)
    out["sdk_ft_cancel"] = cancel_seen and terminal in (
        "cancelled",
        "succeeded",
        "failed",
    )
    try:
        await cl.fine_tuning.jobs.create(model="not-trainable", training_file=f.id)
        out["sdk_ft_untrainable_400"] = False
    except openai.BadRequestError:
        out["sdk_ft_untrainable_400"] = True

    out["sdk_file_delete"] = (await cl.files.delete(f.id)).deleted is True
    await cl.files.delete(bf.id)


async def _probe_vector_stores(cl: Any, out: dict[str, Any]) -> None:
    vs = await cl.vector_stores.create(name="sdk-vs")
    out["sdk_vs_create"] = vs.object == "vector_store" and vs.id.startswith("vs_")
    exp_vs = await cl.vector_stores.create(
        name="sdk-vs-exp",
        extra_body={"expires_after": {"anchor": "last_active_at", "days": 7}},
    )
    out["sdk_vs_expires_after"] = (
        exp_vs.expires_after is not None
        and exp_vs.expires_after.days == 7
        and exp_vs.expires_at is not None
        and exp_vs.last_active_at is not None
    )
    out["sdk_vs_retrieve"] = (await cl.vector_stores.retrieve(vs.id)).id == vs.id
    updated = await cl.vector_stores.update(vs.id, name="sdk-vs-2")
    out["sdk_vs_update"] = updated.name == "sdk-vs-2"
    stores = [s async for s in cl.vector_stores.list(limit=10)]
    out["sdk_vs_list"] = any(s.id == vs.id for s in stores)

    vf_file = await cl.files.create(file=("vs.jsonl", _FT_JSONL), purpose="fine-tune")
    vf = await cl.vector_stores.files.create(vs.id, file_id=vf_file.id)
    out["sdk_vs_file_create"] = vf.id == vf_file.id
    out["sdk_vs_file_retrieve"] = (
        await cl.vector_stores.files.retrieve(vf_file.id, vector_store_id=vs.id)
    ).id == vf_file.id
    vfiles = [x async for x in cl.vector_stores.files.list(vs.id)]
    out["sdk_vs_file_list"] = any(x.id == vf_file.id for x in vfiles)
    vcontent = await cl.vector_stores.files.content(vf_file.id, vector_store_id=vs.id)
    out["sdk_vs_file_content"] = any(
        '"messages"' in c.text for c in vcontent.data if c.type == "text"
    )

    bf_file = await cl.files.create(file=("vsb.jsonl", _FT_JSONL), purpose="fine-tune")
    batch = await cl.vector_stores.file_batches.create(vs.id, file_ids=[bf_file.id])
    out["sdk_vs_batch_create"] = batch.object == "vector_store.files_batch"
    out["sdk_vs_batch_retrieve"] = (
        await cl.vector_stores.file_batches.retrieve(batch.id, vector_store_id=vs.id)
    ).id == batch.id
    bfiles = [
        x async for x in cl.vector_stores.file_batches.list_files(batch.id, vector_store_id=vs.id)
    ]
    out["sdk_vs_batch_files"] = any(x.id == bf_file.id for x in bfiles)

    # the store's search is lexical over file content — query a token the
    # uploaded fixture actually contains (both attached files match, so
    # assert membership, not rank)
    hits = [h async for h in cl.vector_stores.search(vs.id, query="messages")]
    out["sdk_vs_search"] = any(h.file_id == vf_file.id for h in hits)

    vd = await cl.vector_stores.files.delete(vf_file.id, vector_store_id=vs.id)
    out["sdk_vs_file_delete"] = vd.deleted is True and vd.id == vf_file.id
    out["sdk_vs_delete"] = (await cl.vector_stores.delete(vs.id)).deleted is True
    out["sdk_vs_exp_delete"] = (await cl.vector_stores.delete(exp_vs.id)).deleted is True
    await cl.files.delete(vf_file.id)
    await cl.files.delete(bf_file.id)


async def _probe_evals_conversations_uploads(cl: Any, out: dict[str, Any]) -> None:
    spec = await cl.evals.create(
        name="sdk-eval",
        data_source_config={
            "type": "custom",
            "item_schema": {"suite": "calibration", "seed": 0},
        },
        testing_criteria=[{"name": "probe", "type": "label_model"}],
    )
    out["sdk_eval_create"] = spec.id.startswith("eval_") and spec.name == "sdk-eval"
    out["sdk_eval_retrieve"] = (await cl.evals.retrieve(spec.id)).id == spec.id
    evals = [e async for e in cl.evals.list(limit=10)]
    out["sdk_eval_list"] = any(e.id == spec.id for e in evals)
    out["sdk_eval_update"] = (await cl.evals.update(spec.id, name="sdk-eval-2")).name == (
        "sdk-eval-2"
    )
    run = await cl.evals.runs.create(
        spec.id,
        data_source={"type": "custom"},
        extra_body={"model": "fx1"},
    )
    out["sdk_eval_run_create"] = run.id.startswith("evalrun_") and run.eval_id == spec.id
    runs = [rn async for rn in cl.evals.runs.list(spec.id)]
    out["sdk_eval_runs_list"] = any(rn.id == run.id for rn in runs)
    out["sdk_eval_delete"] = (await cl.evals.delete(spec.id)).deleted is True

    conv = await cl.conversations.create(metadata={"lane": "sdk"})
    out["sdk_conv_create"] = conv.id.startswith("conv_") and conv.object == "conversation"
    out["sdk_conv_retrieve"] = (await cl.conversations.retrieve(conv.id)).id == conv.id
    out["sdk_conv_update"] = (
        await cl.conversations.update(conv.id, metadata={"lane": "sdk2"})
    ).metadata["lane"] == "sdk2"
    added = await cl.conversations.items.create(
        conv.id,
        items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
    )
    item_id = added.data[0].id
    out["sdk_conv_items_create"] = len(added.data) == 1 and bool(item_id)
    items = [it async for it in cl.conversations.items.list(conv.id)]
    out["sdk_conv_items_list"] = any(it.id == item_id for it in items)
    one = await cl.conversations.items.retrieve(item_id, conversation_id=conv.id)
    out["sdk_conv_items_retrieve"] = one.id == item_id
    out["sdk_conv_items_delete"] = (
        await cl.conversations.items.delete(item_id, conversation_id=conv.id)
    ).id == conv.id
    out["sdk_conv_delete"] = (await cl.conversations.delete(conv.id)).deleted is True

    up = await cl.uploads.create(
        filename="up.jsonl", bytes=len(_FT_JSONL), mime_type="text/jsonl", purpose="fine-tune"
    )
    out["sdk_upload_create"] = up.id.startswith("upload_") and up.status in (
        "pending",
        "uploaded",
    )
    part = await cl.uploads.parts.create(up.id, data=_FT_JSONL)
    out["sdk_upload_part"] = part.id.startswith("part_")
    done = await cl.uploads.complete(up.id, part_ids=[part.id])
    out["sdk_upload_complete"] = (
        done.status == "completed" and done.file is not None and done.file.id.startswith("file-")
    )
    up2 = await cl.uploads.create(
        filename="up2.jsonl", bytes=8, mime_type="text/jsonl", purpose="fine-tune"
    )
    out["sdk_upload_cancel"] = (await cl.uploads.cancel(up2.id)).status == "cancelled"


async def _probe_embeddings_moderations_errors(cl: Any, out: dict[str, Any]) -> None:
    import openai

    emb = await cl.embeddings.create(model="fx1", input="hello")
    out["sdk_embeddings_create"] = (
        emb.object == "list"
        and emb.data[0].object == "embedding"
        and isinstance(emb.data[0].embedding, list)
        and len(emb.data[0].embedding) > 0
        and emb.usage.total_tokens > 0
    )
    emb64 = await cl.embeddings.create(model="fx1", input=["a", "b"], encoding_format="base64")
    out["sdk_embeddings_base64"] = len(emb64.data) == 2 and all(
        isinstance(d.embedding, str) and len(d.embedding) > 0 for d in emb64.data
    )
    mod = await cl.moderations.create(input="hello", model="omni-moderation-latest")
    out["sdk_moderation"] = (
        bool(mod.results)
        and isinstance(mod.results[0].flagged, bool)
        and mod.results[0].categories is not None
        and mod.results[0].category_scores is not None
    )
    try:
        await cl.chat.completions.create(model="fx1", messages=[])
        out["sdk_err_422_typed"] = False
    except openai.UnprocessableEntityError as exc:
        out["sdk_err_422_typed"] = exc.status_code == 422 and "messages" in str(exc.body)
    try:
        await cl.vector_stores.create(
            name="x", extra_body={"expires_after": {"anchor": "bogus", "days": 0}}
        )
        out["sdk_err_400_typed"] = False
    except openai.BadRequestError as exc:
        out["sdk_err_400_typed"] = exc.status_code == 400
    try:
        await cl.responses.retrieve("resp_nonexistent")
        out["sdk_err_404_typed"] = False
    except openai.NotFoundError as exc:
        out["sdk_err_404_typed"] = exc.status_code == 404


def oai_sdk_audit() -> dict[str, Any]:
    """Run the whole surface under the stock SDK — one event loop, one
    app, probes accumulate as literal bools."""
    out: dict[str, Any] = {}
    loop = asyncio.new_event_loop()
    try:
        with _sdk_client() as cl:
            loop.run_until_complete(_probe_models(cl, out))
            loop.run_until_complete(_probe_chat(cl, out))
            loop.run_until_complete(_probe_completions(cl, out))
            loop.run_until_complete(_probe_responses(cl, out))
            loop.run_until_complete(_probe_files_batches_ft(cl, out))
            loop.run_until_complete(_probe_vector_stores(cl, out))
            loop.run_until_complete(_probe_evals_conversations_uploads(cl, out))
            loop.run_until_complete(_probe_embeddings_moderations_errors(cl, out))
    finally:
        loop.close()
    return out


def oai_sdk_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under oai_sdk_audit.v1."""
    r = oai_sdk_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "oai_sdk_audit",
        "schema": "oai_sdk_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The unmodified openai AsyncOpenAI client drives the harness "
            "end-to-end in-process: models, chat completions (sync + SSE "
            "stream + include_usage), responses (sync + event stream + "
            "cancel), files, batches, fine-tuning jobs incl. "
            "events/checkpoints/cancel, vector stores incl. "
            "expires_after, evals specs+runs, conversations incl. "
            "single-item retrieve, chunked uploads, embeddings, "
            "moderations — and every error lands as the SDK's typed "
            "exception class carrying our error envelope."
            if ok
            else f"OAI SDK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
