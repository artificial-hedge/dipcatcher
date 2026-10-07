"""Stale-quote pickoff: maker age at fill vs forward mid drift.

An aggressive order that lifts a quote submitted long ago is the classic
stale-quote pickoff: the resting order's price no longer reflects the
current information set, so fills on *old* makers should be followed by
larger adverse (aggressor-direction) mid drift than fills on fresh ones.

Both sides are measured exactly: LOBSTER EXECUTION events carry the
resting order's id (linkable to its SUBMISSION time), and the ZI-LOB
`TradeEvent` records `maker_t_submit` directly — no replay needed.

Layers
------
- ``_fill_age_drift`` — core: per-fill (age_s, signed fwd Δmid@h) →
  per-age-bucket mean drift + total count; the pickoff signature is a
  monotone increase of signed drift with maker age.
- ``lobster_stale_quote`` — real tape via the orderbook CSV for mids.
- ``sim_stale_quote`` — sim fills with exact maker ages.
- ``stale_quote_bench`` — real vs sim arms, divergence flags, sealed
  ``stale_quote.v1`` receipt.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    DELETE,
    EXECUTION,
    SUBMISSION,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

AGE_BINS_S = (0.0, 0.1, 0.5, 1.0, 5.0, 30.0, np.inf)
FWD_HORIZON_S = 1.0


def _fill_age_drift(ages: np.ndarray, signed_drift: np.ndarray) -> dict[str, Any]:
    """ages (s) and signed aggressor-direction Δmid at +FWD_HORIZON_S
    (ticks). Bucketed mean drift per age band — monotone increase is the
    stale-quote pickoff signature."""
    n = ages.size
    if n < 50:
        return {"ok": False, "n": int(n)}
    per_bin: list[dict[str, Any]] = []
    for lo, hi in zip(AGE_BINS_S[:-1], AGE_BINS_S[1:], strict=True):
        m = (ages >= lo) & (ages < hi)
        if m.sum() < 5:
            continue
        d = signed_drift[m]
        d = d[np.isfinite(d)]
        per_bin.append(
            {
                "age_lo_s": float(lo),
                "age_hi_s": float(hi) if np.isfinite(hi) else None,
                "n": int(m.sum()),
                "share": float(m.mean()),
                "mean_signed_dmid_ticks": float(np.mean(d)) if d.size else None,
                "mean_size_age_s": float(np.mean(ages[m])),
            }
        )
    finite = np.isfinite(signed_drift)
    return {
        "ok": True,
        "n": int(n),
        "n_with_drift": int(finite.sum()),
        "median_age_s": float(np.median(ages)),
        "share_age_gt_5s": float((ages > 5.0).mean()),
        "mean_signed_dmid_ticks": float(np.mean(signed_drift[finite])) if finite.any() else None,
        "age_bins": per_bin,
    }


def _forward_mids(times: np.ndarray, mids: np.ndarray, horizon: float) -> np.ndarray:
    """mid at first index with time >= t+horizon; NaN where past the end."""
    idx = np.searchsorted(times, times + horizon, side="left")
    out = np.full(times.size, np.nan)
    ok = idx < times.size
    out[ok] = mids[idx[ok]]
    return out


def lobster_stale_quote(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    submit_t: dict[int, float] = {}
    times: list[float] = []
    mids: list[float] = []
    fills: list[tuple[int, float, int]] = []  # (row_idx, t, direction)
    i = 0
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            times.append(ev.time_s)
            mids.append((asks[0][0] + bids[0][0]) / 200.0 if asks and bids else np.nan)
            if ev.event_type == SUBMISSION:
                submit_t[ev.order_id] = ev.time_s
            elif ev.event_type in (EXECUTION,):
                if ev.order_id in submit_t:
                    fills.append((i, ev.time_s - submit_t[ev.order_id], ev.direction))
            elif ev.event_type == DELETE:
                submit_t.pop(ev.order_id, None)
            i += 1
    t_arr = np.asarray(times)
    m_arr = np.asarray(mids)
    fwd = _forward_mids(t_arr, m_arr, FWD_HORIZON_S)
    ages = np.asarray([f[1] for f in fills])
    drift = np.asarray(
        [
            (-f[2]) * (fwd[f[0]] - m_arr[f[0]])
            if np.isfinite(m_arr[f[0]]) and np.isfinite(fwd[f[0]])
            else np.nan
            for f in fills
        ]
    )
    return _fill_age_drift(ages, drift)


def sim_stale_quote(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    n0 = len(sim.trades)
    times: list[float] = []
    mids: list[float] = []
    for _ in range(horizon):
        sim.step()
        times.append(sim.t)
        m = sim.mid
        # sim.mid is in price units; the drift contract is ticks.
        mids.append(float(m) / cfg.tick if m is not None else np.nan)
    t_arr = np.asarray(times, dtype=float)
    m_arr = np.asarray(mids, dtype=float)
    fwd = _forward_mids(t_arr, m_arr, FWD_HORIZON_S)
    ages: list[float] = []
    drifts: list[float] = []
    for tr in sim.trades[n0:]:
        j = int(np.searchsorted(t_arr, tr.t, side="left"))
        if j >= t_arr.size or not np.isfinite(m_arr[j]) or not np.isfinite(fwd[j]):
            continue
        sign = 1.0 if tr.aggressor == "buy" else -1.0
        ages.append(tr.t - tr.maker_t_submit)
        drifts.append(sign * (fwd[j] - m_arr[j]))
    return _fill_age_drift(np.asarray(ages), np.asarray(drifts))


def stale_quote_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_stale_quote(msg_path, ob_path)
    arms = {
        "iid": sim_stale_quote(seed=seed),
        "regime": sim_stale_quote(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_stale_quote(
            SplitFlow(
                p_start=0.10,
                size_tail=1.2,
                k_min=10,
                k_max=600,
                intensity_mult=3.0,
                seed=seed + 2,
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        r = real["share_age_gt_5s"]
        for name, arm in arms.items():
            if arm.get("ok"):
                s = arm["share_age_gt_5s"]
                if abs(r - s) > 0.10:
                    divergences.append(f"{name}_oldfill_{s:.3f}_vs_{r:.3f}")
    payload: dict[str, Any] = {
        "kind": "stale_quote",
        "schema": "stale_quote.v1",
        "ticker": ticker,
        "fwd_horizon_s": FWD_HORIZON_S,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "maker_age_at_fill_times_forward_drift_measured",
        "interpretation": (
            "Each EXECUTION's resting order is linked to its SUBMISSION "
            "time — exact maker age, no replay. signed Δmid at +1s is in "
            "the aggressor's direction: positive = adverse to the maker. "
            "A monotone increase of drift with maker age is the "
            "stale-quote pickoff signature: informed flow preferentially "
            "lifts quotes whose prices went stale. Sim uses "
            "TradeEvent.maker_t_submit directly."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
