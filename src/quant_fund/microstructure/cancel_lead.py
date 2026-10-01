"""cancel_lead — do touch-proximate cancels precede adverse moves?

`cancel_gradient` (#533) showed where liquidity exits (touch-skewed
propensity). This lane asks *why now*: if market makers cancel because
they see the move coming, then mid should drift AWAY from the side
whose quote was just pulled — a bid-side cancel followed by a mid drop
is a dodged fill.

Measurement: for each DELETE/CANCEL_PARTIAL of an order resting at or
within ``TOUCH_BAND_TICKS`` of the own-side touch (pre-event official
row), signed drift = ``side * (mid(t+h) - mid(t))`` for
``h`` in ``HORIZONS_S``. Side is the RESTING side (+1 bid, -1 ask), so
a *negative* signed drift means price moved against the cancelled
order — the cancel was informed. ``np.nan`` tail rows dropped per bin.

Sim arms have no informed agents — cancels fire on an independent
Poisson clock — so signed drift should hover at ~0: the divergence is
the informational content of real withdrawals.

Sealed ``cancel_lead.v1`` receipt, ``data_label: MIXED``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    SUBMISSION,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

TOUCH_BAND_TICKS = 2.0
HORIZONS_S = (0.1, 0.5, 1.0, 5.0)


def _lead_stats(
    signs: np.ndarray,
    canc_t: np.ndarray,
    mid_times: np.ndarray,
    mids: np.ndarray,
    tick_scale: float = 1.0,
) -> dict[str, Any]:
    """Signed post-cancel drift per horizon.

    ``mids`` is sampled at ``mid_times``; each cancel at ``canc_t[i]``
    takes the mid at the first sample >= canc_t[i] (same event row),
    and forward at >= canc_t[i]+h. ``tick_scale`` converts the mid
    units to ticks (1.0 when mids are already in tick units).
    """
    idx = np.searchsorted(mid_times, canc_t, side="left")
    valid = idx < len(mid_times)
    idx, signs = idx[valid], signs[valid]
    m0 = mids[idx]
    per_h: dict[str, Any] = {}
    for h in HORIZONS_S:
        j = np.searchsorted(mid_times, mid_times[idx] + h, side="left")
        ok = j < len(mid_times)
        j_ok = j[ok]
        s_ok = signs[ok]
        drift = s_ok * (mids[j_ok] - m0[ok]) * tick_scale
        drift = drift[np.isfinite(drift)]
        per_h[f"{h}s"] = {
            "n": int(len(drift)),
            "mean_signed_drift_ticks": float(np.mean(drift)) if len(drift) else None,
            "median_signed_drift_ticks": float(np.median(drift)) if len(drift) else None,
            "share_adverse": float(np.mean(drift < 0)) if len(drift) else None,
        }
    return {"ok": len(signs) >= 50, "n_cancels_at_touch": int(len(signs)), "per_horizon": per_h}


def lobster_cancel_lead(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    book: dict[int, tuple[float, int]] = {}  # oid -> (price, side)
    canc_signs: list[float] = []
    canc_times: list[float] = []
    mid_times: list[float] = []
    mids: list[float] = []
    prev_touch: tuple[float, float] | None = None
    with ob_path.open(newline="") as fo:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(fo), strict=True):
            if ev.event_type == SUBMISSION:
                book[ev.order_id] = (float(ev.price), int(ev.direction))
            elif ev.event_type in (DELETE, CANCEL_PARTIAL):
                rec = book.get(ev.order_id)
                if rec is not None and prev_touch is not None:
                    px, side = rec
                    touch = prev_touch[1] if side == 1 else prev_touch[0]
                    dist = (touch - px) / 100.0 if side == 1 else (px - touch) / 100.0
                    if 0.0 <= dist <= TOUCH_BAND_TICKS:
                        canc_signs.append(float(side))
                        canc_times.append(ev.time_s)
                    if ev.event_type == DELETE:
                        book.pop(ev.order_id, None)
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                prev_touch = (float(asks[0][0]), float(bids[0][0]))
                mid_times.append(ev.time_s)
                mids.append((float(asks[0][0]) + float(bids[0][0])) / 200.0)
    if len(canc_signs) < 50:
        return {"ok": False, "n_cancels_at_touch": len(canc_signs)}
    return _lead_stats(
        np.asarray(canc_signs),
        np.asarray(canc_times),
        np.asarray(mid_times),
        np.asarray(mids),
    )


def sim_cancel_lead(
    flow: Any | None = None,
    *,
    seed: int = 0,
    horizon: int = 30000,
    tick: float = 0.01,
) -> dict[str, Any]:
    cfg = ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    canc_signs: list[float] = []
    canc_times: list[float] = []
    mid_times: list[float] = []
    mids: list[float] = []
    for _ in range(horizon):
        before = dict(sim._orders)
        kind = sim.step()
        m = sim.mid
        mid_times.append(float(sim.t))
        mids.append(float(m) if m is not None else np.nan)
        if kind == "cancel":
            ba, bb = sim.best_ask_level, sim.best_bid_level
            for oid in before.keys() - set(sim._orders):
                o = before[oid]
                if o.side == "sell" and ba is not None and 0 <= o.level - ba <= TOUCH_BAND_TICKS:
                    canc_signs.append(-1.0)
                    canc_times.append(float(sim.t))
                elif o.side == "buy" and bb is not None and 0 <= bb - o.level <= TOUCH_BAND_TICKS:
                    canc_signs.append(1.0)
                    canc_times.append(float(sim.t))
    if len(canc_signs) < 50:
        return {"ok": False, "n_cancels_at_touch": len(canc_signs)}
    return _lead_stats(
        np.asarray(canc_signs),
        np.asarray(canc_times),
        np.asarray(mid_times),
        np.asarray(mids),
        tick_scale=1.0 / tick,
    )


def cancel_lead_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_cancel_lead(msg_path, ob_path)
    arms = {
        "iid": sim_cancel_lead(seed=seed),
        "regime": sim_cancel_lead(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_cancel_lead(
            SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    r = real.get("per_horizon", {}).get("1.0s", {}).get("mean_signed_drift_ticks")
    if r is not None:
        for name, arm in arms.items():
            a = arm.get("per_horizon", {}).get("1.0s", {}).get("mean_signed_drift_ticks")
            if a is not None and abs(a - r) > 0.05:
                divergences.append(f"{name}_drift1s_{a:.3f}_vs_{r:.3f}")
    payload: dict[str, Any] = {
        "kind": "cancel_lead",
        "schema": "cancel_lead.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "touch_cancels_lead_mid_real_vs_sim",
        "interpretation": (
            "Cancels within 2 ticks of the own-side touch (pre-event "
            "official row); signed drift = resting_side * (mid(t+h) - "
            "mid(t)) in ticks. Negative mean = price moved against the "
            "cancelled order — informed withdrawal. Positive mean (the "
            "real-tape finding: +0.10 to +0.25 ticks over 0.1-5s, "
            "share_adverse 0.24-0.42) = cancels FOLLOW the move — the "
            "book re-centers and stale quotes get pulled to re-quote "
            "closer (order_revision's cycle). Sim cancels are on an "
            "independent Poisson clock, so drift ~0 is the null."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
