"""Journaled vector stores — the ``/v1/vector_stores`` contract plus the
server-side ``file_search`` retrieval surface for ``/v1/responses``.

Retrieval is lexical, not embedding-backed: a deterministic hashed
bag-of-words cosine scorer (no external embedding dependency, so every
score is reproducible across processes and restarts). Chunks are
word-window slices of the decoded file text; ``file_search`` output
items carry the scored hits verbatim so a caller can audit exactly what
context the model saw — the same injected text lands in the stored
``input_items`` transcript.

Durability rides the same hash-chained ``JobJournal`` as every other
store. The journal records only *identity* events — create, update,
delete, attach, detach — never the derived index. On replay each
attached file is re-read through ``file_reader`` and re-indexed: a file
whose bytes are gone comes back ``status: failed`` with
``last_error.code == 'file_missing_at_replay'``, never as a phantom
searchable record.

The store is transport-free: it raises ``VectorStoreError`` carrying
the HTTP status + wire code; the route layer maps it to ``ApiError``
and the in-process SDK lets it propagate.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from fx1.serve.journal import JobJournal

# Bounds — every cap fails closed, never truncates silently.
VS_MAX_FILES = 32  # attached files per store
VS_MAX_TEXT_BYTES = 4 << 20  # decoded text per file
VS_CHUNK_WORDS = 200  # default window, auto strategy
VS_CHUNK_OVERLAP = 40
VS_MAX_CHUNKS = 512  # indexed windows per file; beyond → truncated=True
VS_MAX_RESULTS = 50  # max_num_results bound on the file_search tool
VS_MAX_QUERY_CHARS = 8192
VS_DIM = 4096  # hashed term space
VS_MAX_ATTRS = 16
VS_ATTR_KEY_MAX = 64
VS_ATTR_VALUE_MAX = 512  # str values
VS_MAX_STORES_ID = 512  # id length bound (path-safe vs_* ids are 36)
VS_FILTER_DEPTH = 4
VS_FILTER_LEAVES = 16
VS_MAX_BATCH_FILES = 500  # file_ids per file_batch (OpenAI's cap)
VS_MAX_EXPIRY_DAYS = 365  # expires_after.days bound (OpenAI's)

_WORD = re.compile(r"[a-z0-9]+")

FILTER_LEAF_TYPES = frozenset({"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"})
FILTER_COMPOUND_TYPES = frozenset({"and", "or"})


class VectorStoreError(Exception):
    """Fail-closed vector-store fault — carries the HTTP status and wire
    code so the route layer can translate verbatim."""

    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def validate_metadata(metadata: dict[str, Any] | None, *, field: str = "metadata") -> None:
    """The shared metadata bound every stamped surface applies."""
    if metadata is None:
        return
    if len(metadata) > 16:
        raise VectorStoreError(400, f"{field} accepts at most 16 entries", "invalid_request")
    for k, v in metadata.items():
        if len(k) > VS_ATTR_KEY_MAX or len(str(v)) > VS_ATTR_VALUE_MAX:
            raise VectorStoreError(
                400,
                f"{field} keys are ≤{VS_ATTR_KEY_MAX} chars, values ≤{VS_ATTR_VALUE_MAX}",
                "invalid_request",
            )


def validate_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """File ``attributes`` — the keys ``filters`` evaluate against.
    Scalars only (str/num/bool); the OpenAI comparison grammar has no
    compound attribute values."""
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise VectorStoreError(400, "attributes must be an object", "invalid_request")
    if len(attributes) > VS_MAX_ATTRS:
        raise VectorStoreError(
            400, f"attributes accepts at most {VS_MAX_ATTRS} entries", "invalid_request"
        )
    for k, v in attributes.items():
        if not isinstance(k, str) or not k or len(k) > VS_ATTR_KEY_MAX:
            raise VectorStoreError(400, "attribute keys are non-empty ≤64 chars", "invalid_request")
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            continue
        if isinstance(v, str) and len(v) <= VS_ATTR_VALUE_MAX:
            continue
        raise VectorStoreError(
            400, "attribute values must be scalar str|num|bool (str ≤512)", "invalid_request"
        )
    return dict(attributes)


def validate_filters(filters: Any, *, depth: int = 0, leaves: list[int] | None = None) -> None:
    """The OpenAI file_search filter grammar: leaf comparisons
    ``{type: eq|ne|gt|gte|lt|lte|in|nin, key, value}`` and compounds
    ``{type: and|or, filters: [...]}``, depth ≤4, ≤16 leaves."""
    if filters is None:
        return
    if not isinstance(filters, dict):
        raise VectorStoreError(400, "filters must be an object", "invalid_filters")
    if depth > VS_FILTER_DEPTH:
        raise VectorStoreError(400, "filters nest at most 4 deep", "invalid_filters")
    leaves = leaves if leaves is not None else [0]
    ftype = filters.get("type")
    if ftype in FILTER_COMPOUND_TYPES:
        subs = filters.get("filters")
        if not isinstance(subs, list) or not subs:
            raise VectorStoreError(
                400, f"filters type {ftype!r} needs a non-empty filters list", "invalid_filters"
            )
        for sub in subs:
            validate_filters(sub, depth=depth + 1, leaves=leaves)
        return
    if ftype not in FILTER_LEAF_TYPES:
        raise VectorStoreError(
            400,
            f"filters.type must be one of {sorted(FILTER_LEAF_TYPES | FILTER_COMPOUND_TYPES)}; "
            f"got {ftype!r}",
            "invalid_filters",
        )
    leaves[0] += 1
    if leaves[0] > VS_FILTER_LEAVES:
        raise VectorStoreError(400, "filters accept at most 16 comparisons", "invalid_filters")
    key = filters.get("key")
    if not isinstance(key, str) or not key or len(key) > VS_ATTR_KEY_MAX:
        raise VectorStoreError(
            400, "filters.key must be a non-empty attribute key", "invalid_filters"
        )
    value = filters.get("value")
    if ftype in ("in", "nin"):
        if not isinstance(value, list) or not value:
            raise VectorStoreError(
                400, f"filters {ftype!r} needs a non-empty value list", "invalid_filters"
            )
        if any(not isinstance(v, (str, int, float, bool)) for v in value):
            raise VectorStoreError(400, "filters value lists carry scalars only", "invalid_filters")
    elif not isinstance(value, (str, int, float, bool)):
        raise VectorStoreError(400, "filters.value must be a scalar", "invalid_filters")


def _filters_match(filters: dict[str, Any] | None, attributes: dict[str, Any]) -> bool:
    """Evaluate a validated filter tree against a file's attributes —
    missing keys fail the leaf (except ``ne``/``nin``, which treat absent
    as not-equal, matching OpenAI's semantics)."""
    if filters is None:
        return True
    ftype = filters["type"]
    if ftype == "and":
        return all(_filters_match(s, attributes) for s in filters["filters"])
    if ftype == "or":
        return any(_filters_match(s, attributes) for s in filters["filters"])
    key = str(filters["key"])
    value = filters["value"]
    have = attributes.get(key)
    if have is None:
        return ftype in ("ne", "nin")
    if ftype == "eq":
        return bool(have == value)
    if ftype == "ne":
        return bool(have != value)
    if ftype == "in":
        return have in value
    if ftype == "nin":
        return have not in value
    if not isinstance(have, (int, float)) or isinstance(have, bool):
        return False  # ordered comparisons need a numeric attribute
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if ftype == "gt":
        return bool(have > value)
    if ftype == "gte":
        return bool(have >= value)
    if ftype == "lt":
        return bool(have < value)
    return bool(have <= value)  # lte


def validate_chunking_strategy(strategy: Any) -> dict[str, Any]:
    """``chunking_strategy`` on attach — ``{"type": "auto"}`` or OpenAI's
    ``{"type": "static", "static": {"max_chunk_size_tokens", "chunk_overlap_tokens"}}``.
    The index is word-based: static bounds map tokens→words at 0.75 and
    land on (window, step); the echoed strategy is verbatim."""
    if strategy is None:
        return {"type": "auto"}
    if not isinstance(strategy, dict):
        raise VectorStoreError(400, "chunking_strategy must be an object", "invalid_request")
    stype = strategy.get("type")
    if stype == "auto":
        return {"type": "auto"}
    if stype != "static":
        raise VectorStoreError(
            400, "chunking_strategy.type must be 'auto' or 'static'", "invalid_request"
        )
    static = strategy.get("static")
    if not isinstance(static, dict):
        raise VectorStoreError(400, "chunking_strategy.static must be an object", "invalid_request")
    size = static.get("max_chunk_size_tokens")
    overlap = static.get("chunk_overlap_tokens", 0)
    if not isinstance(size, int) or isinstance(size, bool) or not 100 <= size <= 4096:
        raise VectorStoreError(
            400, "static.max_chunk_size_tokens must be an int in [100, 4096]", "invalid_request"
        )
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap <= size // 2:
        raise VectorStoreError(
            400,
            "static.chunk_overlap_tokens must be an int in [0, max_chunk_size_tokens/2]",
            "invalid_request",
        )
    return {
        "type": "static",
        "static": {"max_chunk_size_tokens": size, "chunk_overlap_tokens": overlap},
    }


def validate_expires_after(value: Any) -> dict[str, Any] | None:
    """``expires_after`` — OpenAI's anchor policy object. Only
    ``{"anchor": "last_active_at", "days": 1..365}`` exists today;
    anything else fails closed ``invalid_expires_after``."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise VectorStoreError(400, "expires_after must be an object", "invalid_expires_after")
    if set(value) - {"anchor", "days"}:
        raise VectorStoreError(
            400, "expires_after accepts only {anchor, days}", "invalid_expires_after"
        )
    if value.get("anchor") != "last_active_at":
        raise VectorStoreError(
            400,
            "expires_after.anchor must be 'last_active_at'",
            "invalid_expires_after",
        )
    days = value.get("days")
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= VS_MAX_EXPIRY_DAYS:
        raise VectorStoreError(
            400,
            f"expires_after.days must be an int in [1, {VS_MAX_EXPIRY_DAYS}]",
            "invalid_expires_after",
        )
    return {"anchor": "last_active_at", "days": days}


def _strategy_window(strategy: dict[str, Any]) -> tuple[int, int]:
    """(chunk_words, step_words) — static token bounds map to words."""
    if strategy["type"] != "static":
        return VS_CHUNK_WORDS, VS_CHUNK_WORDS - VS_CHUNK_OVERLAP
    size = int(strategy["static"]["max_chunk_size_tokens"])
    overlap = int(strategy["static"]["chunk_overlap_tokens"])
    words = max(8, int(size * 0.75))
    step = max(1, words - int(overlap * 0.75))
    return words, step


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _hash(term: str) -> int:
    return int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "little") % VS_DIM


def _vec(terms: list[str]) -> dict[int, float]:
    counts: dict[int, float] = {}
    for t in terms:
        h = _hash(t)
        counts[h] = counts.get(h, 0.0) + 1.0
    return counts


def _chunk_texts(text: str, strategy: dict[str, Any]) -> tuple[list[str], bool]:
    """Word-window chunking — returns (chunk texts, truncated)."""
    words = text.split()
    if not words:
        return [], False
    window, step = _strategy_window(strategy)
    out: list[str] = []
    i = 0
    while i < len(words) and len(out) < VS_MAX_CHUNKS:
        out.append(" ".join(words[i : i + window]))
        i += step
    return out, i < len(words)


def _cosine(qvec: dict[int, float], cvec: dict[int, float]) -> float:
    """IDF-weighted cosine over the hashed sparse vectors — deterministic,
    no embedding service involved."""
    num = 0.0
    for h, qv in qvec.items():
        num += qv * cvec.get(h, 0.0)
    if not num:
        return 0.0
    qn = math.sqrt(sum(v * v for v in qvec.values()))
    cn = math.sqrt(sum(v * v for v in cvec.values()))
    denom = qn * cn
    return num / denom if denom else 0.0


class VSFileRec(BaseModel):
    """One attached file. ``status`` is terminal at attach time — chunk
    and index run synchronously — so only ``completed``/``failed`` ever
    persist; a replayed file whose bytes vanished re-indexes as
    ``failed`` with ``file_missing_at_replay``."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    vector_store_id: str
    created_at: int
    status: str  # completed | failed
    usage_bytes: int
    filename: str
    attributes: dict[str, Any] = {}
    chunking_strategy: dict[str, Any] = {"type": "auto"}
    indexed_chunks: int = 0
    truncated: bool = False
    last_error: dict[str, Any] | None = None
    text_sha256: str | None = None


class VSBatchRec(BaseModel):
    """One ``file_batch`` — members attach synchronously at create, so
    ``status`` is always terminal (``completed`` when ≥1 file indexed,
    ``failed`` when none did); ``files`` freezes each member's wire row
    at processing time — later detaches don't rewrite the batch's
    verdict."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    vector_store_id: str
    created_at: int
    status: str  # completed | failed (cancelled unreachable: sync attach)
    file_ids: list[str]
    files: list[dict[str, Any]]
    counts: dict[str, int]


class VSMeta(BaseModel):
    """One vector store — ``files`` maps file_id → record (attach order)."""

    model_config = ConfigDict(extra="forbid")

    vs_id: str
    name: str | None
    created_at: int
    metadata: dict[str, Any] = {}
    usage_bytes: int = 0
    files: dict[str, VSFileRec] = {}
    # expiry policy — ``last_active_at`` bumps on attach/batch/search;
    # ``expires_at`` re-derives from it when ``expires_after`` is set
    expires_after: dict[str, Any] | None = None
    last_active_at: int = 0
    expires_at: int | None = None


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    i: int
    text: str
    vec: dict[int, float]


def vs_object(meta: VSMeta) -> dict[str, Any]:
    """The OpenAI ``vector_store`` wire object — ``status`` is
    ``expired`` once ``now >= expires_at`` (the store stays queryable
    for reads; writes/search refuse)."""
    completed = sum(1 for f in meta.files.values() if f.status == "completed")
    failed = sum(1 for f in meta.files.values() if f.status == "failed")
    last_active = meta.last_active_at or meta.created_at
    expired = meta.expires_at is not None and int(time.time()) >= meta.expires_at
    return {
        "id": meta.vs_id,
        "object": "vector_store",
        "created_at": meta.created_at,
        "name": meta.name,
        "status": "expired" if expired else "completed",
        "usage_bytes": meta.usage_bytes,
        "file_counts": {
            "in_progress": 0,
            "completed": completed,
            "cancelled": 0,
            "failed": failed,
            "total": len(meta.files),
        },
        "last_active_at": last_active,
        "expires_after": dict(meta.expires_after) if meta.expires_after else None,
        "expires_at": meta.expires_at,
        "metadata": dict(meta.metadata),
    }


def vs_file_object(rec: VSFileRec) -> dict[str, Any]:
    """The OpenAI ``vector_store.file`` wire object — ``indexed_chunks``
    and ``truncated`` are harness extensions: the honest accounting of
    what the index actually saw."""
    return {
        "id": rec.file_id,
        "object": "vector_store.file",
        "vector_store_id": rec.vector_store_id,
        "created_at": rec.created_at,
        "status": rec.status,
        "usage_bytes": rec.usage_bytes,
        "last_error": rec.last_error,
        "attributes": dict(rec.attributes),
        "chunking_strategy": dict(rec.chunking_strategy),
        "indexed_chunks": rec.indexed_chunks,
        "truncated": rec.truncated,
    }


def vs_batch_object(rec: VSBatchRec) -> dict[str, Any]:
    """The OpenAI ``vector_store.files_batch`` wire object."""
    return {
        "id": rec.batch_id,
        "object": "vector_store.files_batch",
        "created_at": rec.created_at,
        "vector_store_id": rec.vector_store_id,
        "status": rec.status,
        "file_counts": dict(rec.counts),
    }


def _page(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    order: str,
    after: str | None,
    before: str | None,
) -> dict[str, Any]:
    """``{object: list, data, first_id, last_id, has_more}`` — same page
    contract as the stored-request subresources; an unknown cursor fails
    closed ``400 invalid_cursor``."""
    ordered = list(rows)
    if order == "desc":
        ordered.reverse()
    elif order != "asc":
        raise VectorStoreError(
            400, f"order must be 'asc' or 'desc', got {order!r}", "invalid_cursor"
        )
    for cursor, keep_after in ((after, True), (before, False)):
        if cursor is None:
            continue
        idx = next((i for i, it in enumerate(ordered) if it.get("id") == cursor), None)
        if idx is None:
            raise VectorStoreError(
                400, f"cursor {cursor!r} is not in this listing", "invalid_cursor"
            )
        ordered = ordered[idx + 1 :] if keep_after else ordered[:idx]
    page = ordered[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0].get("id") if page else None,
        "last_id": page[-1].get("id") if page else None,
        "has_more": len(ordered) > limit,
    }


FileReader = Any  # Callable[[str], tuple[bytes, str] | None] — bytes + filename


class VectorStoreStore:
    """Bounded journaled store of vector stores + the derived chunk index.

    Bounded by store count (LRU evict, journaled) and by attached files
    per store (fail-closed ``vector_store_full``). Chunks live in memory
    only — the journal never records them; replay rebuilds the index by
    re-reading file bytes through ``file_reader``.
    """

    def __init__(
        self,
        max_stores: int,
        max_files: int = VS_MAX_FILES,
        state_dir: Path | None = None,
        file_reader: FileReader = None,
        *,
        journal: JobJournal | None = None,
    ) -> None:
        if max_stores < 1:
            raise ValueError("max_stores must be at least 1")
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # Compound create-with-files and file-batch operations temporarily
        # pin their store against deletion and capacity eviction. Unrelated
        # stores remain usable while a file reader or index build is slow.
        self._inflight: dict[str, int] = {}
        self._max = max_stores
        self._max_files = max_files
        self._reader = file_reader
        self._stores: OrderedDict[str, VSMeta] = OrderedDict()
        self._chunks: dict[str, dict[str, list[_Chunk]]] = {}
        self._batches: dict[str, dict[str, VSBatchRec]] = {}
        # per-store df + N for the idf table — recomputed lazily
        self._idf_dirty: set[str] = set()
        self._idf: dict[str, dict[int, float]] = {}
        self._journal = (
            journal
            if journal is not None
            else (JobJournal(Path(state_dir) / "vector_stores.jsonl") if state_dir else None)
        )
        self.recover_warnings: list[str] = []
        if self._journal is not None:
            res = self._journal.replay()
            self.recover_warnings = list(res.warnings)
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._drop(str(evict))
                if "vs" in payload:
                    meta = VSMeta.model_validate(payload["vs"])
                    meta.files = {}
                    meta.usage_bytes = 0
                    self._stores[meta.vs_id] = meta
                elif "vs_update" in payload:
                    upd = payload["vs_update"]
                    prev = self._stores.get(str(upd["vs_id"]))
                    if prev is not None:
                        prev.name = upd.get("name")
                        prev.metadata = dict(upd.get("metadata") or {})
                        if "expires_after" in upd:
                            prev.expires_after = upd["expires_after"]
                            prev.expires_at = upd.get("expires_at")
                elif "vs_delete" in payload:
                    self._drop(str(payload["vs_delete"]["vs_id"]))
                elif "vs_touch" in payload:
                    upd = payload["vs_touch"]
                    prev = self._stores.get(str(upd["vs_id"]))
                    if prev is not None:
                        prev.last_active_at = int(upd.get("last_active_at") or 0)
                        prev.expires_at = upd.get("expires_at")
                elif "vs_file" in payload:
                    rec = VSFileRec.model_validate(payload["vs_file"])
                    self._restore_file(rec)
                elif "vs_file_delete" in payload:
                    d = payload["vs_file_delete"]
                    self._detach(str(d["vs_id"]), str(d["file_id"]))
                elif "vs_batch" in payload:
                    brec = VSBatchRec.model_validate(payload["vs_batch"])
                    if brec.vector_store_id in self._stores:
                        self._batches.setdefault(brec.vector_store_id, {})[brec.batch_id] = brec
            self._compact_locked()

    @property
    def max_stores(self) -> int:
        """Configured store-count cap (LRU evicts the oldest at it)."""
        return self._max

    @property
    def max_files(self) -> int:
        """Per-store file-count cap."""
        return self._max_files

    # ---- internals ------------------------------------------------------

    def _drop(self, vs_id: str) -> None:
        self._stores.pop(vs_id, None)
        self._chunks.pop(vs_id, None)
        self._batches.pop(vs_id, None)
        self._idf.pop(vs_id, None)
        self._idf_dirty.discard(vs_id)
        self._inflight.pop(vs_id, None)

    def _pin_locked(self, vs_id: str) -> None:
        self._inflight[vs_id] = self._inflight.get(vs_id, 0) + 1

    def _unpin_locked(self, vs_id: str) -> None:
        remaining = self._inflight.get(vs_id, 0) - 1
        if remaining > 0:
            self._inflight[vs_id] = remaining
        else:
            self._inflight.pop(vs_id, None)
            self._condition.notify_all()

    def _detach(self, vs_id: str, file_id: str) -> None:
        meta = self._stores.get(vs_id)
        rec = meta.files.pop(file_id, None) if meta is not None else None
        if meta is not None and rec is not None:
            meta.usage_bytes = max(0, meta.usage_bytes - rec.usage_bytes)
        self._chunks.get(vs_id, {}).pop(file_id, None)
        self._idf_dirty.add(vs_id)

    def _restore_file(self, rec: VSFileRec) -> None:
        meta = self._stores.get(rec.vector_store_id)
        if meta is None:
            self.recover_warnings.append(
                f"vs file {rec.file_id}: journaled without its store — dropped"
            )
            return
        meta.files[rec.file_id] = rec
        meta.usage_bytes += rec.usage_bytes
        if rec.status != "completed":
            return
        got = self._reader(rec.file_id) if self._reader is not None else None
        if got is None:
            rec.status = "failed"
            rec.indexed_chunks = 0
            rec.truncated = False
            rec.last_error = {
                "code": "file_missing_at_replay",
                "message": "file bytes are gone — record restored, index is empty",
            }
            return
        content, _filename = got
        self._index_file(meta, rec, content)

    def _index_file(self, meta: VSMeta, rec: VSFileRec, content: bytes) -> None:
        """(Re)build the derived chunk index for one attached file."""
        text = content.decode("utf-8", errors="replace")
        texts, truncated = _chunk_texts(text, rec.chunking_strategy)
        if not texts:
            rec.status = "failed"
            rec.last_error = {"code": "empty_file", "message": "file decoded to no text"}
            return
        rec.indexed_chunks = len(texts)
        rec.truncated = truncated
        rec.text_sha256 = hashlib.sha256(content).hexdigest()
        self._chunks.setdefault(meta.vs_id, {})[rec.file_id] = [
            _Chunk(file_id=rec.file_id, i=i, text=t, vec=_vec(_tokens(t)))
            for i, t in enumerate(texts)
        ]
        self._idf_dirty.add(meta.vs_id)

    def _compact_locked(self) -> None:
        if self._journal is None:
            return
        live: list[dict[str, Any]] = []
        for meta in self._stores.values():
            live.append({"vs": meta.model_dump(mode="json", exclude={"files", "usage_bytes"})})
            for rec in meta.files.values():
                live.append({"vs_file": rec.model_dump(mode="json")})
            for batch in self._batches.get(meta.vs_id, {}).values():
                live.append({"vs_batch": batch.model_dump(mode="json")})
        self._journal.compact(live)

    def _store(self, vs_id: str) -> VSMeta:
        if not vs_id or len(vs_id) > VS_MAX_STORES_ID:
            raise VectorStoreError(
                404, f"vector store {vs_id!r} not found", "vector_store_not_found"
            )
        meta = self._stores.get(vs_id)
        if meta is None:
            raise VectorStoreError(
                404, f"vector store {vs_id!r} not found", "vector_store_not_found"
            )
        self._stores.move_to_end(vs_id)
        return meta

    def _expired(self, meta: VSMeta) -> bool:
        """Standing-expiry check — ``expires_at`` passed flips the store
        read-only for writes/search; reads still resolve."""
        return meta.expires_at is not None and int(time.time()) >= meta.expires_at

    def _touch_locked(self, meta: VSMeta) -> None:
        """Activity bump — attaches, batch creates, and searches count
        as use; re-anchors ``expires_at`` when a policy is set."""
        meta.last_active_at = int(time.time())
        if meta.expires_after is not None:
            meta.expires_at = meta.last_active_at + int(meta.expires_after["days"]) * 86400
        if self._journal is not None:
            self._journal.append(
                {
                    "vs_touch": {
                        "vs_id": meta.vs_id,
                        "last_active_at": meta.last_active_at,
                        "expires_at": meta.expires_at,
                    }
                }
            )

    def _idf_table(self, vs_id: str) -> dict[int, float]:
        if vs_id in self._idf_dirty or vs_id not in self._idf:
            chunks_by_file = self._chunks.get(vs_id, {})
            n = sum(len(v) for v in chunks_by_file.values())
            df: dict[int, int] = {}
            for chunks in chunks_by_file.values():
                for c in chunks:
                    for h in c.vec:
                        df[h] = df.get(h, 0) + 1
            self._idf[vs_id] = {h: math.log((1 + n) / (1 + d)) + 1.0 for h, d in df.items()}
            self._idf_dirty.discard(vs_id)
        return self._idf[vs_id]

    # ---- store CRUD ------------------------------------------------------

    def create(
        self,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        file_ids: list[str] | tuple[str, ...] = (),
        expires_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_metadata(metadata)
        policy = validate_expires_after(expires_after)
        created = int(time.time())
        meta = VSMeta(
            vs_id=f"vs_{uuid.uuid4().hex}",
            name=name,
            created_at=created,
            metadata=dict(metadata or {}),
            expires_after=policy,
            last_active_at=created,
            expires_at=created + policy["days"] * 86400 if policy else None,
        )
        with self._condition:
            # Select only unpinned LRU victims. A concurrent compound
            # operation may be reading/indexing files outside the state lock;
            # evicting its store would make that successful request fail 404
            # halfway through.
            evicted: list[str] = []
            while len(self._stores) >= self._max:
                old_id = next(iter(self._stores))
                if self._inflight.get(old_id, 0):
                    self._condition.wait()
                    continue
                self._stores.pop(old_id)
                evicted.append(old_id)
            self._stores[meta.vs_id] = meta
            if self._journal is not None:
                payload: dict[str, Any] = {
                    "vs": meta.model_dump(mode="json", exclude={"files", "usage_bytes"})
                }
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)
            for vid in evicted:
                self._drop(vid)
            if file_ids:
                self._pin_locked(meta.vs_id)
        try:
            for fid in file_ids:
                self.attach(meta.vs_id, fid)
            return vs_object(meta)
        finally:
            if file_ids:
                with self._condition:
                    self._unpin_locked(meta.vs_id)

    def get(self, vs_id: str) -> dict[str, Any]:
        with self._lock:
            return vs_object(self._store(vs_id))

    def update(
        self,
        vs_id: str,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_metadata(metadata)
        policy = validate_expires_after(expires_after)
        with self._lock:
            meta = self._store(vs_id)
            if name is not None:
                meta.name = name
            if metadata is not None:
                meta.metadata = dict(metadata)
            if policy is not None:
                # re-anchor from recorded last activity — an expired
                # store revives honestly here; status recomputes
                meta.expires_after = policy
                meta.expires_at = meta.last_active_at + policy["days"] * 86400
            if self._journal is not None:
                self._journal.append(
                    {
                        "vs_update": {
                            "vs_id": vs_id,
                            "name": meta.name,
                            "metadata": dict(meta.metadata),
                            "expires_after": (
                                dict(meta.expires_after) if meta.expires_after else None
                            ),
                            "expires_at": meta.expires_at,
                        }
                    }
                )
            return vs_object(meta)

    def delete(self, vs_id: str) -> dict[str, Any]:
        with self._condition:
            self._store(vs_id)
            while self._inflight.get(vs_id, 0):
                self._condition.wait()
                self._store(vs_id)
            self._drop(vs_id)
            if self._journal is not None:
                self._journal.append({"vs_delete": {"vs_id": vs_id}})
            return {"id": vs_id, "object": "vector_store.deleted", "deleted": True}

    def list_stores(
        self,
        *,
        limit: int = 20,
        order: str = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            rows = [vs_object(m) for m in self._stores.values()]
            return _page(rows, limit=limit, order=order, after=after, before=before)

    # ---- file membership -------------------------------------------------

    def attach(
        self,
        vs_id: str,
        file_id: str,
        *,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach a ``/v1/files`` record — reads bytes through
        ``file_reader``, chunks + indexes synchronously, then journals
        the membership line. A file that doesn't decode to text lands
        ``status: failed`` honestly instead of refusing."""
        attrs = validate_attributes(attributes)
        strategy = validate_chunking_strategy(chunking_strategy)
        got = self._reader(file_id) if self._reader is not None else None
        with self._lock:
            meta = self._store(vs_id)
            if self._expired(meta):
                raise VectorStoreError(
                    410, f"vector store {vs_id!r} has expired", "vector_store_expired"
                )
            if file_id in meta.files:
                raise VectorStoreError(
                    409, f"file {file_id!r} is already attached", "file_already_attached"
                )
            if len(meta.files) >= self._max_files:
                raise VectorStoreError(
                    409,
                    f"vector store {vs_id!r} holds the {self._max_files}-file cap",
                    "vector_store_full",
                )
            if got is None:
                raise VectorStoreError(404, f"file {file_id!r} not found", "file_not_found")
            content, filename = got
            if len(content) > VS_MAX_TEXT_BYTES:
                raise VectorStoreError(
                    413,
                    f"file exceeds the {VS_MAX_TEXT_BYTES}-byte indexable cap",
                    "file_too_large",
                )
            rec = VSFileRec(
                file_id=file_id,
                vector_store_id=vs_id,
                created_at=int(time.time()),
                status="completed",
                usage_bytes=len(content),
                filename=filename,
                attributes=attrs,
                chunking_strategy=strategy,
            )
            meta.files[file_id] = rec
            meta.usage_bytes += len(content)
            self._index_file(meta, rec, content)
            if self._journal is not None:
                self._journal.append({"vs_file": rec.model_dump(mode="json")})
            self._touch_locked(meta)
            return vs_file_object(rec)

    def list_files(
        self,
        vs_id: str,
        *,
        limit: int = 20,
        order: str = "asc",
        after: str | None = None,
        before: str | None = None,
        filter: str | None = None,
    ) -> dict[str, Any]:
        if filter is not None and filter not in (
            "in_progress",
            "completed",
            "cancelled",
            "failed",
        ):
            raise VectorStoreError(
                400,
                "file filter must be one of in_progress|completed|cancelled|failed",
                "invalid_filters",
            )
        with self._lock:
            meta = self._store(vs_id)
            rows = [
                vs_file_object(r)
                for r in meta.files.values()
                if filter is None or r.status == filter
            ]
            return _page(rows, limit=limit, order=order, after=after, before=before)

    def get_file(self, vs_id: str, file_id: str) -> dict[str, Any]:
        with self._lock:
            meta = self._store(vs_id)
            rec = meta.files.get(file_id)
            if rec is None:
                raise VectorStoreError(
                    404,
                    f"file {file_id!r} is not attached to {vs_id!r}",
                    "file_not_found",
                )
            return vs_file_object(rec)

    def detach(self, vs_id: str, file_id: str) -> dict[str, Any]:
        with self._lock:
            meta = self._store(vs_id)
            rec = meta.files.get(file_id)
            if rec is None:
                raise VectorStoreError(
                    404,
                    f"file {file_id!r} is not attached to {vs_id!r}",
                    "file_not_found",
                )
            meta.files.pop(file_id)
            meta.usage_bytes = max(0, meta.usage_bytes - rec.usage_bytes)
            self._chunks.get(vs_id, {}).pop(file_id, None)
            self._idf_dirty.add(vs_id)
            if self._journal is not None:
                self._journal.append({"vs_file_delete": {"vs_id": vs_id, "file_id": file_id}})
            return {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}

    def file_content(self, vs_id: str, file_id: str) -> dict[str, Any]:
        """``GET .../content`` — the indexed text of one file, chunk by
        chunk (the same strings retrieval can surface)."""
        with self._lock:
            meta = self._store(vs_id)
            rec = meta.files.get(file_id)
            if rec is None:
                raise VectorStoreError(
                    404, f"file {file_id!r} is not attached to {vs_id!r}", "file_not_found"
                )
            chunks = self._chunks.get(vs_id, {}).get(file_id, [])
            return {
                "object": "vector_store.file_content.page",
                "data": [{"type": "text", "text": c.text} for c in chunks],
                "has_more": False,
                "next_page": None,
            }

    # ---- file batches -------------------------------------------------------

    def file_batch_create(
        self,
        vs_id: str,
        file_ids: list[str] | tuple[str, ...],
        *,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST .../file_batches`` — attach many ``file-*`` records in
        one call. Members go through ``attach`` one at a time; a file
        that fails (missing, already attached, oversized, store full)
        counts ``failed`` with its refusal as ``last_error`` — it never
        aborts the batch and never attaches half-validated. Status is
        terminal at return (sync attach): ``completed`` when ≥1 file
        indexed successfully, ``failed`` when none did."""
        if not file_ids or len(file_ids) > VS_MAX_BATCH_FILES:
            raise VectorStoreError(
                400,
                f"file_ids must be a list of 1..{VS_MAX_BATCH_FILES} ids",
                "invalid_request",
            )
        if any(not isinstance(fid, str) or not fid or len(fid) > 128 for fid in file_ids):
            raise VectorStoreError(
                400, "file_ids entries must be non-empty strings ≤128 chars", "invalid_request"
            )
        attrs = validate_attributes(attributes)
        strategy = validate_chunking_strategy(chunking_strategy)
        with self._condition:
            meta = self._store(vs_id)  # ghost store refuses the batch itself, 404
            if self._expired(meta):
                raise VectorStoreError(
                    410, f"vector store {vs_id!r} has expired", "vector_store_expired"
                )
            self._pin_locked(vs_id)
        try:
            created = int(time.time())
            rows: list[dict[str, Any]] = []
            counts = {
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
            for fid in file_ids:
                counts["total"] += 1
                try:
                    rec = self.attach(vs_id, fid, attributes=attrs, chunking_strategy=strategy)
                except VectorStoreError as exc:
                    counts["failed"] += 1
                    rows.append(
                        {
                            "id": fid,
                            "object": "vector_store.file",
                            "vector_store_id": vs_id,
                            "created_at": created,
                            "status": "failed",
                            "usage_bytes": 0,
                            "last_error": {"code": exc.code, "message": str(exc)},
                            "attributes": dict(attrs),
                            "chunking_strategy": dict(strategy),
                            "indexed_chunks": 0,
                            "truncated": False,
                        }
                    )
                else:
                    counts[rec["status"]] += 1
                    rows.append(rec)
            batch = VSBatchRec(
                batch_id=f"vsfb_{uuid.uuid4().hex}",
                vector_store_id=vs_id,
                created_at=created,
                status="completed" if counts["completed"] > 0 else "failed",
                file_ids=list(file_ids),
                files=rows,
                counts=counts,
            )
            with self._lock:
                self._batches.setdefault(vs_id, {})[batch.batch_id] = batch
                if self._journal is not None:
                    self._journal.append({"vs_batch": batch.model_dump(mode="json")})
                self._touch_locked(meta)
            return vs_batch_object(batch)
        finally:
            with self._condition:
                self._unpin_locked(vs_id)

    def _batch(self, vs_id: str, batch_id: str) -> VSBatchRec:
        self._store(vs_id)  # 404 ghost store
        rec = self._batches.get(vs_id, {}).get(batch_id)
        if rec is None:
            raise VectorStoreError(
                404,
                f"file batch {batch_id!r} not found in {vs_id!r}",
                "file_batch_not_found",
            )
        return rec

    def file_batch_get(self, vs_id: str, batch_id: str) -> dict[str, Any]:
        with self._lock:
            return vs_batch_object(self._batch(vs_id, batch_id))

    def file_batch_cancel(self, vs_id: str, batch_id: str) -> dict[str, Any]:
        """``POST .../file_batches/{id}/cancel`` — batches are terminal
        at create (members attach synchronously), so cancel always
        409s with the batch's standing status — honest, never a fake
        mid-flight window."""
        with self._lock:
            rec = self._batch(vs_id, batch_id)
            raise VectorStoreError(
                409,
                f"file batch {batch_id!r} already {rec.status} — "
                "members attach synchronously at create",
                "file_batch_terminal",
            )

    def file_batch_files(
        self,
        vs_id: str,
        batch_id: str,
        *,
        limit: int = 20,
        order: str = "asc",
        after: str | None = None,
        before: str | None = None,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """``GET .../file_batches/{id}/files`` — the frozen per-file
        verdicts in request order (``filter`` takes an OpenAI status
        word)."""
        if filter is not None and filter not in (
            "in_progress",
            "completed",
            "cancelled",
            "failed",
        ):
            raise VectorStoreError(
                400,
                "file filter must be one of in_progress|completed|cancelled|failed",
                "invalid_filters",
            )
        with self._lock:
            rec = self._batch(vs_id, batch_id)
            rows = [dict(r) for r in rec.files if filter is None or r.get("status") == filter]
            return _page(rows, limit=limit, order=order, after=after, before=before)

    # ---- retrieval --------------------------------------------------------

    def search(
        self,
        store_ids: tuple[str, ...] | list[str],
        query: str,
        *,
        max_results: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical search across stores — hashed bag-of-words cosine with
        a per-store idf table. Deterministic: identical corpus + query
        score identically everywhere. ``filters`` apply to file
        attributes (OpenAI's comparison grammar); ``score_threshold``
        drops hits below the cosine bound."""
        if not query or not query.strip():
            raise VectorStoreError(400, "query must be non-empty", "invalid_request")
        if len(query) > VS_MAX_QUERY_CHARS:
            raise VectorStoreError(
                400, f"query exceeds {VS_MAX_QUERY_CHARS} chars", "invalid_request"
            )
        if not 1 <= max_results <= VS_MAX_RESULTS:
            raise VectorStoreError(
                400, f"max_num_results must be in [1, {VS_MAX_RESULTS}]", "invalid_request"
            )
        if score_threshold is not None and not 0.0 <= score_threshold <= 1.0:
            raise VectorStoreError(400, "score_threshold must be in [0, 1]", "invalid_filters")
        validate_filters(filters)
        ids = [str(s) for s in store_ids]
        if not ids:
            raise VectorStoreError(400, "vector_store_ids is empty", "invalid_request")
        qvec_raw = _vec(_tokens(query))
        with self._lock:
            metas = [self._store(s) for s in dict.fromkeys(ids)]
            for meta in metas:
                if self._expired(meta):
                    raise VectorStoreError(
                        410,
                        f"vector store {meta.vs_id!r} has expired",
                        "vector_store_expired",
                    )
            hits: list[dict[str, Any]] = []
            for meta in metas:
                self._touch_locked(meta)
                idf = self._idf_table(meta.vs_id)
                # query vector gets the store's idf applied symmetrically
                qvec = {h: v * idf.get(h, 0.0) for h, v in qvec_raw.items()}
                chunks_by_file = self._chunks.get(meta.vs_id, {})
                for rec in meta.files.values():
                    if rec.status != "completed":
                        continue
                    if not _filters_match(filters, rec.attributes):
                        continue
                    for c in chunks_by_file.get(rec.file_id, []):
                        cvec = {h: v * idf.get(h, 0.0) for h, v in c.vec.items()}
                        score = _cosine(qvec, cvec)
                        if score <= 0.0:
                            continue
                        if score_threshold is not None and score < score_threshold:
                            continue
                        hits.append(
                            {
                                "file_id": rec.file_id,
                                "filename": rec.filename,
                                "vector_store_id": meta.vs_id,
                                "score": score,
                                "text": c.text,
                                "attributes": dict(rec.attributes),
                            }
                        )
            # score desc, then stable ids — deterministic ordering
            hits.sort(key=lambda h: (-float(h["score"]), str(h["file_id"])))
            return hits[:max_results]
