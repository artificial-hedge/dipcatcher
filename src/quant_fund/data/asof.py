"""Fail-closed backward as-of join (SOTA 06 §2.6/§4.3, gap G3).

The as-of join is the primitive that aligns irregular event data onto a regular
decision grid *without* introducing look-ahead — but only when it keys on the
**knowledge** axis (``known_at``/``available_time``, when a fact became
observable) rather than the **event** axis (``event_time``, when it happened in
the world), and only when the publication filter is applied **before** the join.

Every raw ``join_asof`` call site in this repo previously chose its own
``by``/``strategy``/``tolerance`` discipline, which is exactly the
tribal-knowledge pattern LH004's allowlist had to codify. This module is the
single implementation; LH015 flags raw as-of joins that bypass it.

Contract, all of it fail-closed:

1. **Backward only.** There is no ``strategy`` parameter: ``forward`` and
   ``nearest`` pull rows published *after* the decision into the past, so they
   are inexpressible here rather than merely discouraged. (TCA-style
   trade→next-quote matching is a different operation and must not reuse this
   helper.)
2. **Pre-join PIT filter.** ``right_asof`` removes right rows with
   ``right_on > right_asof`` *before* the join runs. Filtering after the join
   is too late — the join itself is the leak vector (SOTA 06 §2.6).
3. **Post-join re-check.** The ``right_asof`` bound is asserted again on the
   joined frame, mirroring ``PitFrame.validate``: a helper bug cannot smuggle a
   future row to the caller.
4. **Inclusive on the knowledge axis by default.** A row published exactly at
   the decision instant *is* knowable at it, so the default boundary is
   ``right_on <= left_on`` (matching ``pit.corrections.select_asof`` and
   ``data.point_in_time.filter_available``). ``boundary="exclusive"`` gives the
   strict ``<`` form, which is what an **event-time** axis needs when
   ``event_time`` stamps a bar *open*: matching a bar that has not closed yet
   reads future-of-the-tick data (SOTA 06 §2.7, AFML ch. 2).
5. **Deterministic ties.** Duplicate ``(by, right_on)`` keys in the visible
   right frame raise. Polars resolves such a tie by arrival order — the same
   bytes in a different order yield a different label — so the ambiguity is
   refused instead of hidden. Collapse versions first (e.g. with
   ``pit.corrections.select_asof``) and join the collapsed frame.
6. **Sortedness is enforced, not assumed.** Polars cannot check sortedness when
   ``by`` groups are present and *silently mis-joins* on unsorted input, so
   both frames are sorted here on ``(by..., key)`` and the caller's left-row
   order is restored on output.
7. **``by`` is mandatory for entity-level frames.** An as-of join without a
   group key across a multi-security frame matches security A's timestamp to
   security B's row — a cross-sectional leak.
8. **Tolerance must be a decision.** ``tolerance`` has no usable default: pass
   a ``timedelta`` to bound staleness (misses become null, never a resurrected
   three-year-old attribute) or ``None`` to accept unbounded staleness
   deliberately, which then records the age of every match in the
   ``{left_on}_staleness`` column so the choice stays visible in the data.

Honesty note: nothing here produces market evidence. The invariant enforced is
a **correctness** invariant (no row published after the decision time can reach
the caller), not a performance claim.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final, Literal

import polars as pl

from quant_fund.schemas.errors import PointInTimeError

__all__ = [
    "ASOF_ENTITY_COLUMNS",
    "Boundary",
    "asof_join",
    "require_asof_timestamp",
]

Boundary = Literal["inclusive", "exclusive"]

#: Column names that identify an entity. Their presence makes ``by`` mandatory:
#: an ungrouped as-of join across entities is a cross-sectional leak.
ASOF_ENTITY_COLUMNS: Final[frozenset[str]] = frozenset(
    {"security_id", "symbol", "ticker", "dataset", "series_id", "instrument_id"}
)

_ROW_ID = "__asof_left_row_id"
_RIGHT_KEY = "__asof_right_key"


def _guard_internal_names(left: pl.DataFrame, right: pl.DataFrame) -> None:
    for frame, what in ((left, "left"), (right, "right")):
        clash = {_ROW_ID, _RIGHT_KEY} & set(frame.columns)
        if clash:
            raise PointInTimeError(
                f"asof_join {what} frame carries reserved internal column(s) {sorted(clash)}"
            )


class _UnsetTolerance:
    """Sentinel: ``tolerance`` was not supplied at all (see module docstring §8)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic aid
        return "<tolerance not specified>"


_UNSET_TOLERANCE = _UnsetTolerance()


def _is_naive(dt: datetime) -> bool:
    return dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None


def require_asof_timestamp(value: datetime, *, what: str) -> datetime:
    """Reject a naive as-of timestamp (fail closed, matching ``pit``)."""
    if not isinstance(value, datetime):
        raise PointInTimeError(f"{what} must be a datetime, got {type(value).__name__}")
    if _is_naive(value):
        raise PointInTimeError(f"{what} must be timezone-aware (UTC); naive stamps are refused")
    return value


def _require_key_column(frame: pl.DataFrame, column: str, *, what: str) -> None:
    if column not in frame.columns:
        raise PointInTimeError(f"{what}: column {column!r} is missing (columns: {frame.columns})")
    dtype = frame.schema[column]
    if not isinstance(dtype, pl.Datetime):
        raise PointInTimeError(
            f"{what}: column {column!r} must be Datetime, got {dtype} — an as-of join on a "
            "non-temporal key cannot establish observability"
        )
    if dtype.time_zone is None:
        raise PointInTimeError(
            f"{what}: column {column!r} must be timezone-aware (UTC), got naive Datetime"
        )
    if frame.height and frame[column].null_count():
        raise PointInTimeError(
            f"{what}: column {column!r} contains null timestamps — a row with no knowledge time "
            "is unobservable and cannot be joined"
        )


def _require_group_columns(left: pl.DataFrame, right: pl.DataFrame, by: tuple[str, ...]) -> None:
    for column in by:
        for frame, what in ((left, "left"), (right, "right")):
            if column not in frame.columns:
                raise PointInTimeError(f"as-of join {what} frame is missing by-column {column!r}")


def _resolve_by(
    left: pl.DataFrame, right: pl.DataFrame, by: str | list[str] | None
) -> tuple[str, ...]:
    if by is not None:
        resolved = (by,) if isinstance(by, str) else tuple(by)
        if not resolved:
            raise PointInTimeError("as-of join `by` must name at least one column")
        _require_group_columns(left, right, resolved)
        return resolved
    entities = sorted(ASOF_ENTITY_COLUMNS & (set(left.columns) | set(right.columns)))
    if entities:
        raise PointInTimeError(
            "as-of join on entity-level frames requires an explicit `by` (found entity "
            f"column(s) {entities}); an ungrouped as-of join would match one entity's "
            "timestamp to another entity's row — a cross-sectional leak"
        )
    return ()


def _visible_right(
    right: pl.DataFrame,
    right_on: str,
    right_asof: datetime | None,
    *,
    inclusive: bool,
) -> pl.DataFrame:
    """Apply the PIT publication filter BEFORE the join (module docstring §2)."""
    if right_asof is None:
        return right
    bound = require_asof_timestamp(right_asof, what="asof_join right_asof")
    predicate = pl.col(right_on) <= pl.lit(bound) if inclusive else pl.col(right_on) < pl.lit(bound)
    visible = right.filter(predicate)
    if visible.height == 0:
        raise PointInTimeError(
            f"as-of join has no right-hand row published by right_asof={bound.isoformat()}; "
            "refusing to join against future knowledge (fail closed)"
        )
    return visible


def _require_unique_keys(right: pl.DataFrame, right_on: str, by: tuple[str, ...]) -> None:
    keys = [*by, right_on]
    if right.select(keys).is_duplicated().any():
        sample = (
            right.select(keys)
            .group_by(keys)
            .len()
            .filter(pl.col("len") > 1)
            .sort("len", descending=True)
            .head(3)
        )
        raise PointInTimeError(
            f"as-of join right frame has duplicate {keys} keys — the match would depend on "
            "row arrival order, so it is refused. Collapse restatements to one version per "
            "key first (e.g. pit.corrections.select_asof). First offenders: "
            f"{sample.to_dicts()}"
        )


def _assert_no_future(
    joined: pl.DataFrame, right_on: str, right_asof: datetime, *, inclusive: bool
) -> None:
    """Defense in depth: re-check the bound on the joined output (§3)."""
    leaked = joined.filter(pl.col(right_on).is_not_null())
    if leaked.height == 0:
        return
    bound = (
        pl.col(right_on) <= pl.lit(right_asof)
        if inclusive
        else pl.col(right_on) < pl.lit(right_asof)
    )
    if leaked.filter(~bound).height:
        raise PointInTimeError(
            f"as-of join produced a row published after right_asof={right_asof.isoformat()} — "
            "future information reached the caller (fail closed)"
        )


def asof_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str,
    right_on: str = "known_at",
    by: str | list[str] | None = None,
    tolerance: timedelta | None | _UnsetTolerance = _UNSET_TOLERANCE,
    right_asof: datetime | None = None,
    boundary: Boundary = "inclusive",
    suffix: str = "_right",
) -> pl.DataFrame:
    """Backward as-of join with point-in-time-safe defaults.

    For each left row, take the most recent right row whose ``right_on`` is at
    (``boundary="inclusive"``, the default) or strictly before
    (``boundary="exclusive"``) the left row's ``left_on``. Rows published after
    ``right_asof`` are removed before the join, so a restatement that became
    known after the decision cannot leak in.

    Args:
        left: Decision grid. Its original row order is preserved on output.
        right: Fact frame. Must carry the knowledge axis in ``right_on`` and be
            unique on ``(by..., right_on)``.
        left_on: Left key column (tz-aware Datetime, no nulls).
        right_on: Right key column; defaults to the knowledge axis ``known_at``.
        by: Group key(s). Mandatory whenever either frame carries an entity
            column (``security_id``, ``symbol``, ...).
        tolerance: ``timedelta`` bounds staleness (older matches become null);
            ``None`` accepts unbounded staleness deliberately and records each
            match's age in ``{left_on}_staleness``. Omitting it is an error.
        right_asof: Decision time for the pre-join publication filter.
        boundary: ``"inclusive"`` for a knowledge axis (published *at* the
            decision instant is knowable), ``"exclusive"`` for an event axis
            whose stamps mark a bar *open* (an unclosed bar is future data).
        suffix: Column suffix for right columns that collide with left ones.

    Returns:
        Left rows in their original order, with the matched right columns
        attached (null where nothing was visible in time).

    Raises:
        PointInTimeError: on missing/naive/null keys, a missing ``by`` on
            entity-level frames, duplicate right keys, no visible right row at
            ``right_asof``, or any post-join bound violation.
        ValueError: on an unknown ``boundary`` or an unspecified ``tolerance``.
    """
    if boundary not in ("inclusive", "exclusive"):
        raise ValueError(f"unknown boundary {boundary!r} (expected 'inclusive' or 'exclusive')")
    if isinstance(tolerance, _UnsetTolerance):
        raise ValueError(
            "asof_join requires an explicit `tolerance`: pass a timedelta to bound staleness, "
            "or None to accept unbounded staleness deliberately (the age of every match is "
            "then recorded in the staleness column)"
        )
    if tolerance is not None and not isinstance(tolerance, timedelta):
        raise TypeError(f"tolerance must be a timedelta or None, got {type(tolerance).__name__}")
    inclusive = boundary == "inclusive"

    _guard_internal_names(left, right)
    _require_key_column(left, left_on, what="asof_join left frame")
    _require_key_column(right, right_on, what="asof_join right frame")
    group_by = _resolve_by(left, right, by)

    if left.height == 0:
        raise PointInTimeError("asof_join left frame is empty; refusing a silent no-op join")
    if right.height == 0:
        raise PointInTimeError(
            "asof_join right frame is empty; refusing an all-null join that would look like "
            "'no information' rather than 'no data'"
        )

    visible = _visible_right(right, right_on, right_asof, inclusive=inclusive)
    _require_unique_keys(visible, right_on, group_by)

    # Sortedness is enforced here, never assumed: polars cannot check it under
    # `by` groups and silently mis-joins unsorted input (module docstring §6).
    prepared_left = left.with_row_index(_ROW_ID).sort([*group_by, left_on, _ROW_ID])
    prepared_right = visible.sort([*group_by, right_on]).rename({right_on: _RIGHT_KEY})

    joined = prepared_left.join_asof(
        prepared_right,
        left_on=left_on,
        right_on=_RIGHT_KEY,
        by=group_by or None,
        strategy="backward",
        allow_exact_matches=inclusive,
        tolerance=tolerance,
        suffix=suffix,
        # Both frames were sorted above; polars cannot check group sortedness
        # under `by` and would only warn, never guarantee.
        check_sortedness=False,
    )

    if right_asof is not None:
        bound = require_asof_timestamp(right_asof, what="asof_join right_asof")
        _assert_no_future(joined, _RIGHT_KEY, bound, inclusive=inclusive)

    if tolerance is None:
        # Unbounded staleness was accepted deliberately; keep the age of every
        # match visible in the data so the decision is auditable (§8).
        joined = joined.with_columns(
            (pl.col(left_on) - pl.col(_RIGHT_KEY)).alias(f"{left_on}_staleness")
        )

    # Output schema matches polars' own as-of join: the right key is carried
    # through under its own name when it differs from the left key, and
    # coalesced into the left key when the names are equal. A right key whose
    # name already exists on the left frame keeps the collision suffix so the
    # left column is never overwritten.
    if right_on == left_on:
        joined = joined.drop(_RIGHT_KEY)
    elif right_on in left.columns:
        joined = joined.rename({_RIGHT_KEY: f"{right_on}{suffix}"})
    else:
        joined = joined.rename({_RIGHT_KEY: right_on})

    # Restore the caller's left-row order; drop the internal bookkeeping column.
    return joined.sort(_ROW_ID).drop(_ROW_ID)
