"""Flatten a book before a published listing end.

The event-loop backtest marks a held name from the last print it has, then
raises ``StaleValuationError`` once that print ages out. A next-open fill
cannot exit on a session that does not exist. When the security master has
already published ``valid_to``, the book has to be flat on the prior decision
so the exit fills on the last print.

``valid_to`` is usable only at decisions on or after the master row's
``available_time``. An end that is not yet published, or that falls outside
the sample, is left alone. A name that is already flat is left alone. A held
name whose end is inside the sample but was not published in time to exit
raises ``ListingExitError`` instead of carrying a stale position.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

_FLAT = 1e-12


class ListingExitError(ValueError):
    """A published listing end cannot be exited on a session that still prints."""


def apply_published_listing_exits(
    weights: pl.DataFrame,
    bars: pl.DataFrame,
    root: str | Path,
) -> pl.DataFrame:
    """Apply ``schedule_listing_exits`` when bronze ``security_master.parquet`` exists."""
    path = Path(root) / "bronze" / "security_master.parquet"
    if not path.is_file():
        return weights
    master = pl.read_parquet(path)
    if master.is_empty() or "valid_to" not in master.columns:
        return weights
    return schedule_listing_exits(weights, bars, master)


def schedule_listing_exits(
    weights: pl.DataFrame,
    bars: pl.DataFrame,
    master: pl.DataFrame,
) -> pl.DataFrame:
    """Zero targets so each in-sample published listing end exits on its last print."""
    _require_columns(weights, ("event_time", "security_id", "target_weight"), "weights")
    _require_columns(bars, ("event_time", "security_id"), "bars")
    _require_columns(master, ("security_id", "valid_to", "available_time"), "security master")
    if weights.is_empty():
        raise ListingExitError("weights are empty")
    if bars.is_empty():
        raise ListingExitError("bars are empty")
    ended = master.filter(pl.col("valid_to").is_not_null()).sort("security_id")
    if ended.is_empty():
        return weights
    panel_end = _as_datetime(bars.get_column("event_time").max(), "panel", "event_time")
    out = weights
    changed = False
    for row in ended.iter_rows(named=True):
        nxt = _flatten_one(out, bars, row, panel_end)
        if nxt is not out:
            changed = True
            out = nxt
    if changed:
        return out.sort(["event_time", "security_id"])
    return weights


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ListingExitError(f"{what} missing columns: {missing}")


def _as_datetime(value: object, sid: str, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    raise ListingExitError(f"{sid}: {field} must be a datetime")


def _flatten_one(
    weights: pl.DataFrame,
    bars: pl.DataFrame,
    row: dict[str, object],
    panel_end: datetime,
) -> pl.DataFrame:
    sid = str(row["security_id"])
    valid_to = _as_datetime(row["valid_to"], sid, "valid_to")
    available = _as_datetime(row["available_time"], sid, "available_time")
    listed = bars.filter((pl.col("security_id") == sid) & (pl.col("event_time") < valid_to))
    if listed.is_empty():
        raise ListingExitError(f"{sid}: published listing end but no print before it")
    last_print = _as_datetime(listed.get_column("event_time").max(), sid, "event_time")
    if valid_to > panel_end or last_print >= panel_end:
        return weights
    decisions = [
        stamp
        for stamp in weights.get_column("event_time").unique().sort().to_list()
        if isinstance(stamp, datetime) and stamp < last_print
    ]
    if _already_flat(weights, sid, decisions, last_print):
        return weights
    eligible = [stamp for stamp in decisions if available <= stamp]
    if not eligible:
        raise ListingExitError(
            f"{sid}: listing ends at {valid_to.isoformat()} before a published exit can fill"
        )
    prior = eligible[-1]
    updated = _zero_from(weights, sid, prior)
    held = weights.filter((pl.col("security_id") == sid) & (pl.col("event_time") == prior))
    if held.is_empty():
        updated = _append_flat(updated, sid, prior)
    return updated


def _already_flat(
    weights: pl.DataFrame,
    sid: str,
    decisions: list[datetime],
    last_print: datetime,
) -> bool:
    carried = _carried_weight(weights, sid, decisions[-1]) if decisions else 0.0
    later = weights.filter((pl.col("security_id") == sid) & (pl.col("event_time") >= last_print))
    later_gross = float(later.get_column("target_weight").abs().sum()) if later.height else 0.0
    return abs(carried) <= _FLAT and later_gross <= _FLAT


def _carried_weight(weights: pl.DataFrame, sid: str, when: datetime) -> float:
    rows = weights.filter((pl.col("security_id") == sid) & (pl.col("event_time") <= when)).sort(
        "event_time"
    )
    if rows.is_empty():
        return 0.0
    return float(rows.get_column("target_weight").tail(1).item())


def _zero_from(weights: pl.DataFrame, sid: str, when: datetime) -> pl.DataFrame:
    return weights.with_columns(
        pl.when((pl.col("security_id") == sid) & (pl.col("event_time") >= when))
        .then(pl.lit(0.0))
        .otherwise(pl.col("target_weight"))
        .alias("target_weight")
    )


def _append_flat(weights: pl.DataFrame, sid: str, when: datetime) -> pl.DataFrame:
    sample = weights.head(1)
    overrides: list[pl.Expr] = [
        pl.lit(when).cast(weights.schema["event_time"]).alias("event_time"),
        pl.lit(sid).cast(weights.schema["security_id"]).alias("security_id"),
        pl.lit(0.0).cast(weights.schema["target_weight"]).alias("target_weight"),
    ]
    for name, dtype in weights.schema.items():
        if name in {"event_time", "security_id", "target_weight"}:
            continue
        overrides.append(pl.lit(None).cast(dtype).alias(name))
    return pl.concat([weights, sample.with_columns(overrides)], how="vertical")
