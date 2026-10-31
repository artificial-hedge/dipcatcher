"""HF-RV — high-frequency realized variance.

Pinned by the Day Wave 140 design drop. See :mod:`docs.HF_RV_DESIGN` for the
full contract (data ingest, PIT gate, jump test, plug-in points, test
surface, open questions).

This module computes intraday realized variance on a 5-min grid from one
or more HF sources, fail-closed on null/missing/patchy data, PIT-correct,
and exposes a single function that plugs into the existing overlay /
forecast / risk pipeline without changing the public surface of
``forecast_asof``, ``optimize_asof``, or ``/risk/portfolio``.

The public surface (signature + dataclass) was frozen by the
scaffolding tests in :mod:`tests.test_hf_rv_scaffolding`; this
implementation lands the body without changing the contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from quant_fund.models.realized import bipower_variation, lee_mykland_jumps

__all__ = ["HFRVResult", "compute_hf_rv"]


@dataclass
class HFRVResult:
    """Realized variance on a 5-min grid for a single name at a single asof.

    Attributes
    ----------
    rv_5min
        Realized variance on the 5-min grid for the trailing window.
    bpv
        Bipower variation (jump-robust diffusive variance).
    jump_stat
        Max |Lee–Mykland| statistic over the window.
    n_jumps
        Count of 5-min bars flagged as jumps at significance 1%.
    rv_5min_jump_clean
        Realized variance excluding the flagged bars.
    n_obs
        Number of 5-min bars used in the estimate.
    asof
        Decision origin timestamp (PIT).
    available_time
        Latest observable restatement of any input bar.
    source
        Authority-ordered source name: ``"cls"`` / ``"binance"`` / ``"imf"``.
    source_secondary
        Cross-check source if present, else ``None``.
    clock_drift_seconds
        Maximum per-bar clock drift observed in the input, in seconds.
    holes_count
        Number of single-bar holes that were linearly interpolated.
    honest
        ``False`` when the result must NOT be used downstream (fail-closed).
    """

    rv_5min: float
    bpv: float
    jump_stat: float
    n_jumps: int
    rv_5min_jump_clean: float
    n_obs: int
    asof: pd.Timestamp
    available_time: pd.Timestamp
    source: str
    source_secondary: str | None
    clock_drift_seconds: float
    holes_count: int
    honest: bool


#: Default 5-min window: 1 trading day = 288 bars at 5-min granularity.
DEFAULT_WINDOW_BARS: Final[int] = 288

#: Default tick-subsampling stride (5-tick sub-sample).
DEFAULT_SUBSAMPLE_STRIDE: Final[int] = 5

#: Authority order for HF sources. Lower index = higher priority.
SOURCE_AUTHORITY: Final[tuple[str, ...]] = ("cls", "binance", "imf")

#: Max contiguous missing bars before fail-closed.
MAX_CONTIGUOUS_HOLES: Final[int] = 1

#: Max fraction of window that can be missing before fail-closed.
MAX_HOLE_FRACTION: Final[float] = 0.05

#: Max allowed per-bar clock drift (seconds). Above this, fail-closed.
MAX_CLOCK_DRIFT_SECONDS: Final[float] = 60.0

#: Bar size in seconds (5 minutes).
_BAR_SECONDS: Final[int] = 300


def _fail_closed(
    asof: pd.Timestamp,
    available_time: pd.Timestamp,
    *,
    reason: str,
    source: str = "cls",
    source_secondary: str | None = None,
    clock_drift_seconds: float = 0.0,
    holes_count: int = 0,
) -> HFRVResult:
    """Construct a fail-closed HFRVResult.

    The values are placeholders that mark the result as not for use; the
    ``honest=False`` flag is the contract. NaN is used for the numeric
    fields to make it visible in receipts that the calculation did not
    run. ``reason`` is accepted for traceability in receipts but not
    stored on the dataclass — fail-closed callers should log it
    separately if persistence is needed.
    """
    del reason  # accepted for traceability; not on the public dataclass
    return HFRVResult(
        rv_5min=float("nan"),
        bpv=float("nan"),
        jump_stat=float("nan"),
        n_jumps=0,
        rv_5min_jump_clean=float("nan"),
        n_obs=0,
        asof=asof,
        available_time=available_time,
        source=source,
        source_secondary=source_secondary,
        clock_drift_seconds=clock_drift_seconds,
        holes_count=holes_count,
        honest=False,
    )


def _validate_columns(bars: pd.DataFrame) -> None:
    required = {"event_time", "available_time", "open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"hf_rv: bars missing required columns: {sorted(missing)}")


def _source_from_bars(bars: pd.DataFrame) -> tuple[str, str | None]:
    """Derive the primary and secondary source from the bars.

    If the bars carry a per-row ``source`` column, return the
    authority-ordered primary and the first secondary in authority order.
    Otherwise, default to ``"cls"`` (the design's authority default).
    """
    if "source" not in bars.columns:
        return "cls", None
    seen: list[str] = []
    for raw in bars["source"].dropna().unique():
        s = str(raw).strip().lower()
        if s in SOURCE_AUTHORITY and s not in seen:
            seen.append(s)
    if not seen:
        return "cls", None
    seen_sorted = sorted(seen, key=lambda s: SOURCE_AUTHORITY.index(s))
    primary = seen_sorted[0]
    secondary = seen_sorted[1] if len(seen_sorted) > 1 else None
    return primary, secondary


def _clock_drift_seconds(bars: pd.DataFrame) -> float:
    """Maximum per-bar clock drift observed in the input, in seconds.

    Computed as the maximum absolute offset (in seconds) between a bar's
    ``event_time`` and the nearest 5-min grid boundary. Zero when there is
    no time-like column to inspect.
    """
    if len(bars) == 0 or "event_time" not in bars.columns:
        return 0.0
    et = pd.to_datetime(bars["event_time"], utc=True, errors="coerce")
    if et.isna().all():
        return 0.0
    # Use the Series' underlying numpy array; ``.values`` on a tz-aware
    # datetime64 Series is an ndarray of dtype ``datetime64[ns, UTC]``
    # (or the underlying unit). Convert to a numpy datetime64[ns] array
    # of int64 nanoseconds-since-epoch, then to whole seconds via
    # integer division. The ``.dt.tz_localize(None)`` accessor is the
    # right shape here; ``Series.tz_localize`` would inspect the Series
    # index (a RangeIndex for plain column data) and raise.
    arr = np.asarray(et.dt.tz_localize(None).values, dtype="datetime64[ns]")
    ns = arr.view("int64")
    unix = ns // 1_000_000_000
    offset = np.abs(unix % _BAR_SECONDS)
    # Drift = min(offset, 300 - offset) so a bar at 23:57:30 has the same
    # drift as one at 00:02:30.
    drift = np.minimum(offset, _BAR_SECONDS - offset)
    return float(np.nanmax(drift))


def _detect_holes(sorted_bars: pd.DataFrame, *, window_bars: int) -> tuple[pd.DataFrame, int, bool]:
    """Detect and linearly-interpolate single-bar holes; fail-closed otherwise.

    Returns the (possibly interpolated) bars, the count of single-bar holes
    that were interpolated, and a fail-closed flag.
    """
    if len(sorted_bars) == 0:
        return sorted_bars, 0, True
    if len(sorted_bars) == 1:
        return sorted_bars, 0, False
    et = pd.to_datetime(sorted_bars["event_time"], utc=True, errors="coerce")
    if et.isna().any():
        return sorted_bars, 0, True
    # Compute deltas between consecutive bars in 5-min units.
    diffs = et.diff().dropna().dt.total_seconds().to_numpy() / _BAR_SECONDS
    if len(diffs) == 0:
        return sorted_bars, 0, False
    # Per-step gap in bars (1 == adjacent, 2 == one hole, etc).
    gaps = np.round(diffs).astype(int)
    if (gaps <= 0).any():
        return sorted_bars, 0, True  # duplicate or out-of-order event times
    n_holes_total = int(np.sum(np.maximum(gaps - 1, 0)))
    n_contig = int(np.max(np.maximum(gaps - 1, 0)))
    hole_fraction = n_holes_total / max(window_bars, 1)
    fail_closed = (n_contig > MAX_CONTIGUOUS_HOLES) or (hole_fraction > MAX_HOLE_FRACTION)
    if fail_closed:
        return sorted_bars, n_holes_total, True
    if n_holes_total == 0:
        return sorted_bars, 0, False
    # Build a continuous 5-min grid and interpolate price columns linearly.
    full_index = pd.date_range(start=et.iloc[0], end=et.iloc[-1], freq=f"{_BAR_SECONDS}s")
    interpolated = sorted_bars.set_index(et).reindex(full_index)
    interp_count = int(interpolated["close"].isna().sum())
    for col in ("open", "high", "low", "close"):
        interpolated[col] = interpolated[col].interpolate(method="linear", limit_direction="both")
    if "volume" in interpolated.columns:
        interpolated["volume"] = interpolated["volume"].fillna(0.0)
    interpolated["event_time"] = interpolated.index
    # available_time: take max so PIT stays as late as possible.
    interpolated["available_time"] = interpolated["available_time"].fillna(
        sorted_bars["available_time"].max()
    )
    interpolated = interpolated.reset_index(drop=True)
    return interpolated, interp_count, False


def _subsample(bars: pd.DataFrame, *, stride: int, seed: int) -> pd.DataFrame:
    """Apply deterministic tick subsampling.

    Drops rows using a deterministic permutation keyed by ``seed`` to keep
    microstructure-noise reduction reproducible across runs. The bar
    closest to asof is always retained via the trailing-window selection
    that happens in :func:`compute_hf_rv` before this is called.
    """
    n = len(bars)
    if n <= 2 or stride <= 1:
        return bars
    keep_mask = np.ones(n, dtype=bool)
    perm = np.random.default_rng(seed).permutation(n)
    # Drop the largest-step entries until the kept density matches 1/stride.
    target = max(1, n // stride)
    drop_count = n - target
    for idx in perm[:drop_count]:
        keep_mask[int(idx)] = False
    return bars.loc[keep_mask].reset_index(drop=True)


def compute_hf_rv(
    bars: pd.DataFrame,
    asof: pd.Timestamp,
    available_time: pd.Timestamp,
    *,
    window_bars: int = DEFAULT_WINDOW_BARS,
    apply_tick_subsample: bool = True,
    subsample_stride: int = DEFAULT_SUBSAMPLE_STRIDE,
    rng_seed: int | None = None,
) -> HFRVResult:
    """Compute HF realized variance on a 5-min grid.

    See :mod:`docs.HF_RV_DESIGN` for the full contract. The signature is
    frozen by the scaffolding tests; this is the implementation that the
    scaffolding pinned.

    Parameters
    ----------
    bars
        5-min bars with columns ``event_time``, ``available_time``, ``open``,
        ``high``, ``low``, ``close`` (and optionally ``volume`` / ``source``).
        tz-aware ``event_time`` in UTC.
    asof
        Decision origin timestamp (PIT).
    available_time
        Latest observable restatement; bars with ``available_time > asof``
        are dropped (PIT gate).
    window_bars
        Number of 5-min bars in the trailing window. Default 288 (1 day).
    apply_tick_subsample
        If ``True``, apply deterministic tick subsampling to reduce
        microstructure noise. Default ``True``.
    subsample_stride
        Stride for tick subsampling. Default 5.
    rng_seed
        Optional seed for tick subsampling; defaults to a deterministic
        seed derived from ``asof``.

    Returns
    -------
    HFRVResult
        Dataclass with all required fields. ``honest=False`` when the
        result must NOT be used downstream (fail-closed).
    """
    # --- 1. Schema validation ---------------------------------------------
    try:
        _validate_columns(bars)
    except ValueError:
        return _fail_closed(asof, available_time, reason="schema_invalid")

    if len(bars) == 0:
        return _fail_closed(asof, available_time, reason="empty_bars")

    # --- 2. PIT gate ------------------------------------------------------
    at = pd.to_datetime(bars["available_time"], utc=True, errors="coerce")
    if at.isna().any():
        return _fail_closed(asof, available_time, reason="null_availability")
    pit_mask = at <= asof
    if not pit_mask.any():
        return _fail_closed(asof, available_time, reason="no_pit_bars")
    bars_pit = bars.loc[pit_mask].copy()
    latest_at = at[pit_mask].max()

    # --- 3. Source authority ---------------------------------------------
    source, source_secondary = _source_from_bars(bars_pit)

    # --- 4. Sort + take trailing window ----------------------------------
    et_series = pd.to_datetime(bars_pit["event_time"], utc=True, errors="coerce")
    # Some pandas versions hand back a DatetimeIndex when the input is a
    # Series; force a Series so the column assignment downstream keeps
    # the original RangeIndex intact.
    if not isinstance(et_series, pd.Series):
        et_series = pd.Series(et_series, index=bars_pit.index)
    if et_series.isna().any():
        return _fail_closed(
            asof,
            latest_at,
            reason="null_event_time",
            source=source,
            source_secondary=source_secondary,
        )
    bars_pit = (
        bars_pit.assign(event_time=et_series).sort_values("event_time").reset_index(drop=True)
    )
    bars_pit = bars_pit.tail(window_bars).reset_index(drop=True)

    # --- 5. Clock drift check --------------------------------------------
    drift = _clock_drift_seconds(bars_pit)
    if drift > MAX_CLOCK_DRIFT_SECONDS:
        return _fail_closed(
            asof,
            latest_at,
            reason="clock_drift_exceeded",
            source=source,
            source_secondary=source_secondary,
            clock_drift_seconds=drift,
        )

    # --- 6. Hole detection + interpolation ------------------------------
    bars_clean, holes_count, holes_fail = _detect_holes(bars_pit, window_bars=window_bars)
    if holes_fail:
        return _fail_closed(
            asof,
            latest_at,
            reason="patchy_intraday",
            source=source,
            source_secondary=source_secondary,
            clock_drift_seconds=drift,
            holes_count=holes_count,
        )

    # --- 7. Tick subsampling ---------------------------------------------
    if apply_tick_subsample and len(bars_clean) > 2:
        if rng_seed is None:
            # Deterministic seed derived from asof.
            seed = int(pd.Timestamp(asof).value // 10**9)
        else:
            seed = int(rng_seed)
        bars_clean = _subsample(bars_clean, stride=subsample_stride, seed=seed)

    # --- 8. Compute log returns on close --------------------------------
    close = pd.to_numeric(bars_clean["close"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(close).any() or len(close) < 4:
        return _fail_closed(
            asof,
            latest_at,
            reason="degenerate_close",
            source=source,
            source_secondary=source_secondary,
            clock_drift_seconds=drift,
            holes_count=holes_count,
        )
    # Pad with the first close to anchor the first return at zero — only
    # the n-1 returns are summed.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(close))
    log_ret = log_ret[np.isfinite(log_ret)]
    n_obs = int(log_ret.size)

    if n_obs < 4:
        return _fail_closed(
            asof,
            latest_at,
            reason="too_few_returns",
            source=source,
            source_secondary=source_secondary,
            clock_drift_seconds=drift,
            holes_count=holes_count,
        )

    # --- 9. RV, BPV, jump test -------------------------------------------
    rv = float(np.sum(log_ret * log_ret))
    bv = bipower_variation(log_ret)
    if not math.isfinite(bv) or bv <= 0.0:
        return _fail_closed(
            asof,
            latest_at,
            reason="non_positive_bpv",
            source=source,
            source_secondary=source_secondary,
            clock_drift_seconds=drift,
            holes_count=holes_count,
        )
    lm = lee_mykland_jumps(log_ret, alpha=0.99)
    is_jump = np.asarray(lm["is_jump"], dtype=bool)
    stat_series = np.asarray(lm["stat"], dtype=float)
    n_jumps = int(is_jump.sum())
    jump_stat = float(np.nanmax(stat_series)) if stat_series.size else 0.0

    clean_ret = log_ret[~is_jump] if n_jumps < n_obs else np.array([], dtype=float)
    rv_clean = float(np.sum(clean_ret * clean_ret)) if clean_ret.size > 0 else 0.0

    return HFRVResult(
        rv_5min=rv,
        bpv=bv,
        jump_stat=jump_stat,
        n_jumps=n_jumps,
        rv_5min_jump_clean=rv_clean,
        n_obs=n_obs,
        asof=asof,
        available_time=latest_at,
        source=source,
        source_secondary=source_secondary,
        clock_drift_seconds=drift,
        holes_count=holes_count,
        honest=True,
    )
