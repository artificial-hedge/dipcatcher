"""Signal sleeves for the perp/spot books — each emits causal target weights.

Every function returns ``(event_time, security_id, target_weight)`` frames
consumable by ``run_perp_backtest`` / ``run_backtest``. All rolling statistics
are shift-guarded: a weight decided at bar t only sees data from bars strictly
before t's close, so fills at t+1 open are causally sound.

These produce *weight proposals* — leverage, margin and risk-gate enforcement
live in the engines, not here.

Window/holding-period arguments require Python integers, not booleans or floats.
Volatility windows require >=2 observations. Funding lookbacks require >=1
(>=10 for spike-fade's fixed warmup). Hysteresis ``vol_lookback`` requires >=5
or None, preserving its five-return warmup. Momentum permits ``skip_bars=0``
(current-close momentum); lookback must be positive and exceed the skip.
Trend means accept ``1 <= fast_bars < slow_bars``. Sweep lookback requires >=2
and hold length >=1. Residual windows require >=2; sigma warmup is the smaller
of the z window and max(4, z_window // 4), so short windows use all observations.
"""

from __future__ import annotations

from bisect import insort
from datetime import datetime

import numpy as np
import polars as pl

from quant_fund.northset.sweeps import liquidity_sweep_frame

_WCOLS = ("event_time", "security_id", "target_weight")


def _validate_window(value: int, label: str, minimum: int) -> None:
    """Reject coercion and boolean counts before rolling/shift operations."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}")


def _validate_bars(bars: pl.DataFrame) -> None:
    missing = {"event_time", "security_id", "close"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")


def _cap_and_emit(
    frame: pl.DataFrame,
    raw_col: str,
    *,
    max_name: float,
    gross_scale: float,
) -> pl.DataFrame:
    """Clip names to ±max_name, scale the cross-section to gross_scale.

    ``gross_scale`` bounds sum(|w|) per timestamp — the engine's leverage cap
    still applies on top.
    """
    out = (
        frame.filter(pl.col(raw_col).is_not_null() & pl.col(raw_col).is_finite())
        .with_columns(
            (pl.col(raw_col).clip(-float(max_name), float(max_name))).alias("target_weight")
        )
        .with_columns(
            pl.when(pl.col("target_weight").abs().sum().over("event_time") > gross_scale)
            .then(
                pl.col("target_weight")
                * gross_scale
                / pl.col("target_weight").abs().sum().over("event_time")
            )
            .otherwise(pl.col("target_weight"))
            .alias("target_weight")
        )
        .filter(pl.col("target_weight").abs() > 1e-9)
        .select("event_time", "security_id", "target_weight")
    )
    return out.sort(["event_time", "security_id"])


def _per_symbol_vol(bars: pl.DataFrame, window: int) -> pl.DataFrame:
    """Causal per-bar volatility estimate: rolling std of log returns, shifted."""
    _validate_bars(bars)
    _validate_window(window, "vol_window", 2)
    return (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("close").log().diff().over("security_id").alias("_r"),
        )
        .with_columns(
            pl.col("_r")
            .rolling_std(window, min_samples=max(2, window // 4))
            .shift(1)
            .over("security_id")
            .alias("_vol")
        )
        .select("security_id", "event_time", "_vol")
    )


def _join_available_funding(
    grid: pl.DataFrame,
    funding: pl.DataFrame,
    lookback_events: int,
    *,
    min_samples: int = 1,
) -> pl.DataFrame:
    """Join trailing event-ordered statistics of observations known at each bar.

    Availability order may differ from event order. Insert newly eligible
    events into a bounded event-ordered window rather than delaying a window
    containing unknown events, or treating a late old print as the latest one.
    Event-only inputs retain the legacy event-time availability convention.
    Explicit availability is mandatory when the column is supplied.
    """
    required = {"security_id", "event_time", "value"}
    missing = required - set(funding.columns)
    if missing:
        raise ValueError(f"funding missing columns: {sorted(missing)}")
    _validate_window(lookback_events, "lookback_events", min_samples)
    available = "available_time" if "available_time" in funding.columns else "event_time"
    for column in {"event_time", available}:
        if not isinstance(funding.schema[column], pl.Datetime) or funding[column].null_count():
            raise ValueError(f"funding {column} must contain non-null datetimes")
    if funding["security_id"].null_count():
        raise ValueError("funding security_id must be non-null")
    if funding["value"].null_count() or not funding["value"].is_finite().all():
        raise ValueError("funding value must be finite and non-null")
    if funding.select("security_id", "event_time").is_duplicated().any():
        raise ValueError("funding has duplicate (security_id, event_time) observations")
    events = funding.select(
        "security_id",
        pl.col("event_time").dt.cast_time_unit("ns").cast(pl.Int64).alias("_event"),
        pl.col(available).dt.cast_time_unit("ns").cast(pl.Int64).alias("_available"),
        "value",
    ).with_columns(pl.max_horizontal("_event", "_available").alias("_funding_time"))
    snapshots = []
    for group in events.partition_by("security_id", maintain_order=True):
        window: list[tuple[int, float]] = []
        for sid, event, _, value, eligible in group.sort(["_funding_time", "_event"]).iter_rows():
            insort(window, (event, value))
            if len(window) > lookback_events:
                window.pop(0)
            values = np.asarray([rate for _, rate in window], dtype=float)
            mean = float(values.mean()) if len(window) >= min_samples else None
            std = float(values.std(ddof=1)) if len(window) >= max(2, min_samples) else None
            z = (
                (window[-1][1] - mean) / (std + 1e-8)
                if mean is not None and std is not None
                else None
            )
            snapshots.append((sid, eligible, mean, z))
    history = (
        pl.DataFrame(
            snapshots,
            schema={
                "security_id": funding.schema["security_id"],
                "_funding_time": pl.Int64,
                "rate_ma": pl.Float64,
                "_z": pl.Float64,
            },
            orient="row",
        )
        .unique(subset=["security_id", "_funding_time"], keep="last")
        .sort("_funding_time")
    )
    return (
        grid.with_columns(
            pl.col("event_time").dt.cast_time_unit("ns").cast(pl.Int64).alias("_funding_time")
        )
        .sort("_funding_time")
        .join_asof(history, on="_funding_time", by="security_id", strategy="backward")
        .drop("_funding_time")
        .sort(["security_id", "event_time"])
    )


def funding_carry_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    lookback_events: int = 3,
    max_name: float = 0.05,
    gross_scale: float = 2.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Carry sleeve: long symbols whose funding is negative (shorts pay you),
    short symbols with positive funding — sized by rate z-score, inverse-vol.

    Uses only funding events realized and available by the decision bar
    (event_time <= t and available_time <= t, inclusive). ``lookback_events`` averages the last N
    realized rates (~N×8h of history on Binance's standard grid).
    """
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    if "value" not in funding.columns:
        raise ValueError("funding frame needs a 'value' column (rate)")
    grid = bars.select("security_id", "event_time", "close").sort(["security_id", "event_time"])
    vols = _per_symbol_vol(bars, vol_window)
    joined = (
        _join_available_funding(grid, funding, lookback_events)
        .join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(-pl.col("rate_ma") / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns((pl.col("_raw") - pl.col("_raw").median().over("event_time")).alias("_raw"))
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def funding_spike_fade_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    lookback_events: int = 30,
    z_threshold: float = 2.0,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Crowded-positioning fade: symbols whose funding rate is an extreme
    outlier vs their own history are over-levered on one side — fade them
    (short extreme-positive funding, long extreme-negative). Only acts beyond
    ``z_threshold``; between events the last extreme print's weight is carried
    flat until another realized funding observation becomes available."""
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    grid = bars.select("security_id", "event_time", "close").sort(["security_id", "event_time"])
    vols = _per_symbol_vol(bars, vol_window)
    joined = (
        _join_available_funding(grid, funding, lookback_events, min_samples=10)
        .join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_z").abs() >= z_threshold)
            .then(
                -pl.col("_z").sign()
                / pl.when(pl.col("_vol") > 0).then(pl.col("_vol")).otherwise(None)
            )
            .otherwise(0.0)
            .alias("_raw")
        )
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def cross_sectional_momentum_weights(
    bars: pl.DataFrame,
    *,
    lookback_bars: int = 168,
    skip_bars: int = 4,
    max_name: float = 0.05,
    gross_scale: float = 2.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Momentum sleeve: cross-sectional rank of return over
    ``[t-lookback, t-skip]`` (skip the freshest bars to blunt reversal),
    demeaned across the universe each bar, inverse-vol sized."""
    _validate_bars(bars)
    _validate_window(lookback_bars, "lookback_bars", 1)
    _validate_window(skip_bars, "skip_bars", 0)
    if lookback_bars <= skip_bars:
        raise ValueError("lookback_bars must exceed skip_bars")
    base = (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close").shift(skip_bars) / pl.col("close").shift(lookback_bars) - 1.0)
            .over("security_id")
            .alias("_mom")
        )
        .drop_nulls("_mom")
    )
    vols = _per_symbol_vol(bars, vol_window)
    frame = (
        base.join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            (
                pl.col("_mom").rank().over("event_time") / pl.col("_mom").count().over("event_time")
                - 0.5
            ).alias("_rank")
        )
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(pl.col("_rank") / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns(pl.col("_raw") - pl.col("_raw").median().over("event_time"))
    )
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def sweep_reclaim_weights(
    bars: pl.DataFrame,
    *,
    lookback: int = 20,
    hold_bars: int = 8,
    decay: float = 0.75,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
) -> pl.DataFrame:
    """Dipcatch sleeve: a swept-and-reclaimed prior extreme is a stop-run
    exhaustion — long after low-sweep reclaims, short after high-sweep
    reclaims. The pulse decays geometrically over ``hold_bars``."""
    _validate_bars(bars)
    _validate_window(lookback, "lookback", 2)
    _validate_window(hold_bars, "hold_bars", 1)
    if not 0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    sweeps = liquidity_sweep_frame(bars, lookback=lookback).select(
        "security_id", "event_time", "sweep_reject_signed"
    )
    # Geometric hold: emit a decaying pulse for the next hold_bars bars.
    parts: list[pl.DataFrame] = []
    for h in range(hold_bars):
        parts.append(
            sweeps.select(
                "security_id",
                pl.col("event_time").alias("signal_time"),
                (pl.col("sweep_reject_signed") * (decay**h)).alias("_raw"),
            )
        )
        parts[-1] = parts[-1].with_columns(_h=pl.lit(h))
    pulses = pl.concat(parts).filter(pl.col("_raw") != 0.0)
    # Map each signal to bar index offsets per symbol.
    bar_idx = bars.sort(["security_id", "event_time"]).with_columns(
        pl.int_range(0, pl.len()).over("security_id").alias("_i")
    )
    sig = pulses.join(
        bar_idx.select("security_id", "event_time", "_i"),
        left_on=["security_id", "signal_time"],
        right_on=["security_id", "event_time"],
        how="inner",
    ).with_columns((pl.col("_i") + pl.col("_h")).alias("_ti"))
    placed = sig.join(
        bar_idx.select("security_id", pl.col("_i").alias("_ti"), "event_time"),
        on=["security_id", "_ti"],
        how="inner",
    )
    frame = (
        placed.group_by(["security_id", "event_time"])
        .agg(pl.col("_raw").sum())
        .sort(["security_id", "event_time"])
    )
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def slow_trend_weights(
    bars: pl.DataFrame,
    *,
    fast_bars: int = 168,
    slow_bars: int = 720,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Slow trend sleeve: sign(fast mean − slow mean) × inverse-vol,
    demeaned cross-sectionally so the book is roughly dollar-neutral."""
    _validate_bars(bars)
    _validate_window(fast_bars, "fast_bars", 1)
    _validate_window(slow_bars, "slow_bars", 2)
    if fast_bars >= slow_bars:
        raise ValueError("fast_bars must be < slow_bars")
    base = (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            (
                pl.col("close")
                .rolling_mean(fast_bars, min_samples=fast_bars)
                .shift(1)
                .over("security_id")
                - pl.col("close")
                .rolling_mean(slow_bars, min_samples=slow_bars)
                .shift(1)
                .over("security_id")
            ).alias("_spread")
        )
        .drop_nulls("_spread")
    )
    vols = _per_symbol_vol(bars, vol_window)
    frame = (
        base.join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(pl.col("_spread").sign() / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns(pl.col("_raw") - pl.col("_raw").median().over("event_time"))
    )
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def basis_carry_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    lookback_events: int = 3,
    max_name: float = 0.15,
    gross_scale: float = 0.9,
    min_rate: float = 0.0,
) -> pl.DataFrame:
    """Delta-neutral carry weights: allocate the pair book across symbols
    whose trailing realized funding is positive, proportional to the rate.

    Weight ``w_i ∝ max(rate_ma_i − min_rate, 0)`` normalized to sum≈1 across
    qualifying names per timestamp, then per-name capped and gross-scaled.
    Trailing windows include only events realized and available by the bar.
    Only positive rates are eligible — negative-funding harvest needs spot
    borrow, which the carry book does not model.
    """
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    if "value" not in funding.columns:
        raise ValueError("funding frame needs a 'value' column (rate)")
    grid = bars.select("security_id", "event_time").sort(["security_id", "event_time"])
    joined = (
        _join_available_funding(grid, funding, lookback_events)
        .with_columns(
            pl.when(pl.col("rate_ma") > min_rate)
            .then(pl.col("rate_ma") - min_rate)
            .otherwise(0.0)
            .alias("_raw")
        )
        .with_columns(
            pl.when(pl.col("_raw").sum().over("event_time") > 0)
            .then(pl.col("_raw") / pl.col("_raw").sum().over("event_time"))
            .otherwise(0.0)
            .alias("_raw")
        )
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def basis_carry_hysteresis_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    enter_rate: float = 0.0003,
    exit_rate: float = 0.0,
    lookback_events: int = 3,
    name_weight: float = 0.08,
    max_names: int = 10,
    rebalance_band: float | None = None,
    rate_scale_ref: float | None = None,
    rate_scale_cap: float = 2.0,
    rate_scale_floor: float = 0.3,
    vol_lookback: int | None = None,
    vol_ref: float = 0.04,
    rate_exponent: float = 0.0,
    enter_rate_by_prefix: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Event-driven carry membership book — the low-churn version.

    A symbol enters when its trailing mean realized funding ≥ ``enter_rate``
    and exits when it falls below ``exit_rate``; between membership changes no
    weight rows are emitted, so the engine holds the pair untouched (no daily
    rebalancing churn). Slots are capped at ``max_names``; contested slots go
    to the highest realized rates at each decision timestamp. Funding enters
    trailing windows only once both event and availability times have passed
    (equality is eligible).

    ``rebalance_band`` bounds notional drift: a held pair's weight grows with
    price (fixed units), so when its mark drifts outside
    ``[name_weight / band, name_weight * band]`` a resize row re-emitting
    ``name_weight`` is emitted — sparse, only on breach. Without it, positions
    entered cheap can drift to many multiples of the intended weight and blow
    past the perp book's leverage/margin limits (the engine caps leverage only
    at order time). ``None`` keeps the original hold-untouched behaviour.

    ``rate_scale_ref`` enables regime-scaled sizing: when set, all weight rows
    emitted at a timestamp are multiplied by
    ``clip(book_rate / rate_scale_ref, rate_scale_floor, rate_scale_cap)``
    where ``book_rate`` is the mean trailing rate across the day's qualifying
    candidates (top ``max_names``) plus currently held names — the book
    self-sizes with funding dispersion (deploys more when yields are rich,
    thins when they compress) without adding churn between event days.

    ``vol_lookback`` enables per-name vol scaling: each emitted weight is also
    multiplied by ``min(1, vol_ref / vol_i)`` where ``vol_i`` is the symbol's
    daily close-close return std over the last ``vol_lookback`` bars. Wildest
    microcaps — the ones that drive basis-squeeze drawdowns — get diluted
    weight while calm names keep full size.

    ``rate_exponent`` tilts weights toward higher payers: each name's weight
    is multiplied by ``N * rate_i**rate_exponent / sum(rate_j**rate_exponent)``
    over the day's book, so the book's gross is unchanged but share follows
    rate^exponent. ``0`` = equal weight (default).

    ``enter_rate_by_prefix`` overrides ``enter_rate`` for sids carrying the
    given prefix (e.g. ``{"DYDX:": 0.0006}``) — venues with noisier funding
    prints can demand deeper persistence before membership; exit stays global.
    """
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    if vol_lookback is not None:
        _validate_window(vol_lookback, "vol_lookback", 5)
    if rebalance_band is not None and not (np.isfinite(rebalance_band) and rebalance_band > 1.0):
        raise ValueError("rebalance_band must be > 1 or None")
    grid = _join_available_funding(
        bars.select("security_id", "event_time", "close"), funding, lookback_events
    )
    rates_by_time: dict[datetime, dict[str, float]] = {}
    px_by_time: dict[datetime, dict[str, float]] = {}
    for row in grid.iter_rows(named=True):
        rates_by_time.setdefault(row["event_time"], {})[str(row["security_id"])] = row["rate_ma"]
        px_by_time.setdefault(row["event_time"], {})[str(row["security_id"])] = row["close"]

    _enter_bars: dict[str, float] = {}
    if enter_rate_by_prefix:
        known = {str(s) for s in funding["security_id"].unique()}
        for sid in known:
            for pref, bar in enter_rate_by_prefix.items():
                if sid.startswith(pref):
                    _enter_bars[sid] = bar
                    break

    vols_by_time: dict[datetime, dict[str, float]] = {}
    if vol_lookback is not None:
        vroll = (
            bars.select("security_id", "event_time", "close")
            .sort(["security_id", "event_time"])
            .with_columns(
                pl.col("close")
                .pct_change()
                .rolling_std(vol_lookback, min_samples=5)
                .over("security_id")
                .alias("vol")
            )
            .select("security_id", "event_time", "vol")
        )
        for row in vroll.iter_rows(named=True):
            vols_by_time.setdefault(row["event_time"], {})[str(row["security_id"])] = row["vol"]

    active: dict[str, float] = {}
    entry_px: dict[str, float] = {}
    out_t: list[datetime] = []
    out_s: list[str] = []
    out_w: list[float] = []
    for t in sorted(rates_by_time):
        rates = rates_by_time[t]
        px_now = px_by_time.get(t, {})
        for sid in list(active):
            r = rates.get(sid)
            if r is None or r < exit_rate:
                out_t.append(t)
                out_s.append(sid)
                out_w.append(0.0)
                del active[sid]
                entry_px.pop(sid, None)
        cands = sorted(
            (
                (sid, r)
                for sid, r in rates.items()
                if r is not None and r >= _enter_bars.get(sid, enter_rate) and sid not in active
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        scale = 1.0
        if rate_scale_ref is not None and rate_scale_ref > 0:
            book_rates = [r for _, r in cands[:max_names]] + [
                rates[s] for s in active if rates.get(s) is not None
            ]
            if book_rates:
                scale = min(
                    rate_scale_cap,
                    max(rate_scale_floor, (sum(book_rates) / len(book_rates)) / rate_scale_ref),
                )
        w_now = name_weight * scale
        tilts: dict[str, float] = {}
        if rate_exponent > 0:
            book: dict[str, float | None] = {sid: r for sid, r in cands[:max_names]}
            for sid in active:
                book.setdefault(sid, rates.get(sid))
            rates_pos = {s: r for s, r in book.items() if isinstance(r, (int, float)) and r > 0}
            if rates_pos:
                n_book = max(len(book), 1)
                norm = n_book / sum(r**rate_exponent for r in rates_pos.values())
                tilts = {s: norm * r**rate_exponent for s, r in rates_pos.items()}
        vols_t = vols_by_time.get(t, {})

        def _vscale(sid: str, w: float, vols_t: dict[str, float] = vols_t) -> float:
            v = vols_t.get(sid)
            if v is None or v <= 0:
                return w
            return w * min(1.0, vol_ref / v)

        def _w(sid: str, w_now: float = w_now, tilts: dict[str, float] = tilts) -> float:
            return _vscale(sid, w_now * tilts.get(sid, 1.0))

        if rebalance_band is not None:
            for sid in list(active):
                p0 = entry_px.get(sid)
                p1 = px_now.get(sid)
                if p0 is None or p1 is None or not (p0 > 0 and p1 > 0):
                    continue
                drift = p1 / p0
                if drift >= rebalance_band or drift <= 1.0 / rebalance_band:
                    out_t.append(t)
                    out_s.append(sid)
                    out_w.append(_w(sid))
                    entry_px[sid] = p1
        for sid, _r in cands:
            if len(active) >= max_names:
                break
            w_sid = _w(sid)
            active[sid] = w_sid
            pe = px_now.get(sid)
            if pe is not None and pe > 0:
                entry_px[sid] = pe
            out_t.append(t)
            out_s.append(sid)
            out_w.append(w_sid)
    return pl.DataFrame({"event_time": out_t, "security_id": out_s, "target_weight": out_w}).sort(
        ["event_time", "security_id"]
    )


def residual_mr_weights(
    bars: pl.DataFrame,
    *,
    factor_window: int = 96,
    z_window: int = 48,
    reversal_window: int = 4,
    z_clip: float = 3.0,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
) -> pl.DataFrame:
    """Kakushadze-style cross-sectional residual mean-reversion.

    The per-timestamp equal-weight mean return is the book factor; each name's
    beta to it comes from a trailing covariance/variance window, residuals are
    z-scored against their own trailing sigma, and the signal is the negative
    of the clipped z summed over ``reversal_window`` — long residual-oversold
    names, short residual-overbought, dollar-neutral after demeaning.

    Strictly causal: beta, sigma and the residual feed are all shifted so the
    emitted weight at row ``t`` depends only on bars strictly before ``t``.
    """
    _validate_bars(bars)
    for label, v in (
        ("factor_window", factor_window),
        ("z_window", z_window),
        ("reversal_window", reversal_window),
    ):
        _validate_window(v, label, 2)
    if z_clip <= 0.0 or not np.isfinite(z_clip):
        raise ValueError("z_clip must be positive and finite")

    frame = bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("close").log().diff().over("security_id").alias("_r")
    )
    frame = frame.with_columns(pl.col("_r").mean().over("event_time").alias("_mkt"))
    # Trailing per-name beta to the book factor, all estimators shifted by one
    # bar so no current-bar return enters them. Match the population covariance
    # E[r*m] - E[r]E[m] with population variance (ddof=0) in the denominator.
    frame = frame.with_columns(
        (
            (
                (pl.col("_r") * pl.col("_mkt")).rolling_mean(factor_window)
                - pl.col("_r").rolling_mean(factor_window)
                * pl.col("_mkt").rolling_mean(factor_window)
            )
            / pl.col("_mkt").rolling_var(factor_window, ddof=0)
        )
        .shift(1)
        .over("security_id")
        .alias("_beta")
    )
    # Residual uses the trailing beta on the observed return, then the whole
    # residual feed is shifted once more so the z-signal at t reads <= t-1.
    frame = frame.with_columns(
        (pl.col("_r") - pl.col("_beta") * pl.col("_mkt")).alias("_resid_now")
    )
    frame = frame.with_columns(
        pl.col("_resid_now").shift(1).over("security_id").alias("_resid"),
        pl.col("_resid_now")
        .shift(2)
        .rolling_std(z_window, min_samples=min(z_window, max(4, z_window // 4)))
        .over("security_id")
        .alias("_resid_sigma"),
    )
    frame = frame.with_columns(
        (
            (-(pl.col("_resid") / pl.col("_resid_sigma")))
            .clip(-float(z_clip), float(z_clip))
            .rolling_mean(reversal_window, min_samples=1)
        )
        .over("security_id")
        .alias("_raw")
    )
    # Dollar-neutral cross-section: subtract the per-timestamp median.
    frame = frame.with_columns(pl.col("_raw") - pl.col("_raw").median().over("event_time"))
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def blend_weights(
    sleeves: dict[str, pl.DataFrame],
    sleeve_weights: dict[str, float],
) -> pl.DataFrame:
    """Sum sleeve weight panels under named multipliers, then re-net
    duplicates (sleeve netting happens here — one weight per (t, sid))."""
    parts = []
    for name, frame in sleeves.items():
        mult = float(sleeve_weights.get(name, 0.0))
        if not np.isfinite(mult):
            raise ValueError(f"sleeve weight for {name} must be finite")
        if mult == 0.0:
            continue
        parts.append(frame.with_columns((pl.col("target_weight") * mult).alias("target_weight")))
    if not parts:
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime,
                "security_id": pl.String,
                "target_weight": pl.Float64,
            }
        )
    return (
        pl.concat(parts)
        .group_by(["event_time", "security_id"])
        .agg(pl.col("target_weight").sum())
        .sort(["event_time", "security_id"])
    )
