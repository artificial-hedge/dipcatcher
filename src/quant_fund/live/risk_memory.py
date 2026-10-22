"""Persist the pre-trade rings the gateway must replay across CLI processes.

Rate, duplicate, session-start, and peak memory live on the engine. A fresh
process used to construct an empty engine, so those limits reset. This module
copies the live ring entries out and back without re-deciding the orders.
"""

from __future__ import annotations

from typing import Any

from quant_fund.live.errors import LiveGateError
from quant_fund.pretrade.engine import PretradeEngine
from quant_fund.pretrade.kernel import _note_duplicate


def export_pretrade(engine: PretradeEngine) -> dict[str, Any]:
    """Snapshot the counters a later process has to enforce."""
    book = engine.book
    return {
        "session_start": float(book.session_start),
        "peak": float(book.peak),
        "last_ts": int(book.last_ts),
        "symbols": list(engine._ids),
        "order_ts": _stamps(book.ots, int(book.o_tail), int(book.o_live), int(book.ring_cap)),
        "message_ts": _stamps(book.mts, int(book.m_tail), int(book.m_live), int(book.ring_cap)),
        "dupes": _pairs(
            book.r_fp, book.r_ts, int(book.r_tail), int(book.r_live), int(book.ring_mask)
        ),
    }


def restore_pretrade(engine: PretradeEngine, blob: object) -> None:
    """Put a snapshot back onto a new engine. Malformed memory fails closed."""
    if not isinstance(blob, dict):
        raise LiveGateError("stored pre-trade memory is malformed")
    book = engine.book
    symbols = blob.get("symbols")
    if not isinstance(symbols, list):
        raise LiveGateError("stored pre-trade symbols are malformed")
    for symbol in symbols:
        if not isinstance(symbol, str) or engine.ensure_symbol(symbol) < 0:
            raise LiveGateError("stored pre-trade symbol could not be restored")
    try:
        book.session_start = float(blob["session_start"])
        book.peak = float(blob["peak"])
        book.last_ts = int(blob["last_ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveGateError("stored pre-trade floors are malformed") from exc
    engine._recompute_floors()
    _restore_ring(
        book.ots,
        _ints(blob.get("order_ts"), "order timestamps"),
        int(book.ring_cap),
        "order",
    )
    book.o_tail = 0
    book.o_live = len(_ints(blob.get("order_ts"), "order timestamps"))
    book.o_head = book.o_live
    messages = _ints(blob.get("message_ts"), "message timestamps")
    _restore_ring(book.mts, messages, int(book.ring_cap), "message")
    book.m_tail = 0
    book.m_live = len(messages)
    book.m_head = book.m_live
    for fingerprint, stamp in _pairs_from(blob.get("dupes")):
        _note_duplicate(book, fingerprint, stamp)


def _stamps(slots: list[int], tail: int, live: int, cap: int) -> list[int]:
    found: list[int] = []
    index = tail
    for _ in range(live):
        found.append(int(slots[index]))
        index += 1
        if index == cap:
            index = 0
    return found


def _pairs(
    fingerprints: list[int],
    stamps: list[int],
    tail: int,
    live: int,
    mask: int,
) -> list[list[int]]:
    found: list[list[int]] = []
    index = tail
    for _ in range(live):
        found.append([int(fingerprints[index]), int(stamps[index])])
        index = (index + 1) & mask
    return found


def _ints(raw: object, label: str) -> list[int]:
    if not isinstance(raw, list):
        raise LiveGateError(f"stored pre-trade {label} are malformed")
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise LiveGateError(f"stored pre-trade {label} are malformed") from exc


def _restore_ring(slots: list[int], values: list[int], cap: int, label: str) -> None:
    if len(values) > cap:
        raise LiveGateError(f"stored pre-trade {label} ring does not fit")
    for index, stamp in enumerate(values):
        slots[index] = stamp


def _pairs_from(raw: object) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        raise LiveGateError("stored pre-trade duplicates are malformed")
    pairs: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise LiveGateError("stored pre-trade duplicates are malformed")
        try:
            pairs.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError) as exc:
            raise LiveGateError("stored pre-trade duplicates are malformed") from exc
    return pairs
