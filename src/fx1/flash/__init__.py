"""Flash context — persistent research memory for the fx-1 front door.

Findings are stored with their sources, timestamps, task relevance,
uncertainty, and conflicting evidence in a local JSONL store under
``$FX1_CONFIG_DIR/flash/`` (default ``~/.fx1/flash/``). Retrieval is a
deterministic hashed-vector + recency scoring pass — no embedding service,
no network — so the relevant portions of prior research reach the current
task while unrelated material is left behind.

Public surface:
- ``store`` — the entry schema and the append-only :class:`FlashStore`.
- ``retrieve`` — relevance scoring and context-block assembly.
- ``refresh`` — re-check an entry's sources and mark staleness honestly.
"""

from fx1.flash.refresh import RefreshReport, refresh, refresh_entry
from fx1.flash.retrieve import (
    Retrieval,
    context_block,
    retrieve,
    score_entry,
)
from fx1.flash.store import (
    FlashEntry,
    FlashSource,
    FlashStore,
    add,
    default_store,
    get,
    list_entries,
    remove,
    update,
)

__all__ = [
    "FlashEntry",
    "FlashSource",
    "FlashStore",
    "RefreshReport",
    "Retrieval",
    "add",
    "context_block",
    "default_store",
    "get",
    "list_entries",
    "refresh",
    "refresh_entry",
    "remove",
    "retrieve",
    "score_entry",
    "update",
]
