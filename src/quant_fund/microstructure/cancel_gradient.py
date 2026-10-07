"""Cancel gradient: cancellation propensity vs distance from the touch.

Where does resting liquidity exit? If cancels concentrate at the touch,
the top of the book churns while depth sits — market makers skim the
queue. If cancels are uniform over depth, exits are inventory-driven.

For every DELETE/CANCEL_PARTIAL the resting order's price is known
(id-linked from its SUBMISSION), and its distance from the own-side
touch at cancel time comes from the prior orderbook row. Propensity =
share of cancels in bucket d / share of resting volume in bucket d:
>1 means that depth band exits disproportionately.

Layers
------
- ``_gradient`` — per-distance-bucket cancel count share / depth share.
- ``lobster_cancel_gradient`` — real tape, both sides.
- ``sim_cancel_gradient`` — oid-diff on ``sim._orders``; levels in ticks.
- ``cancel_gradient_bench`` — divergences + sealed ``cancel_gradient.v1``.
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
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

# distance buckets in ticks: 0 = at touch, 1..3, 4..10, 11..50, >50
BINS = (0, 1, 4, 11, 51)
BIN_LABELS = ("touch", "d1_3", "d4_10", "d11_50", "d51_plus")


def _gradient(cancel_d: np.ndarray, depth_d: np.ndarray, depth_w: np.ndarray) -> dict[str, Any]:
    """cancel_d: distance-at-cancel per event; depth_d/depth_w: sampled
    resting volume by distance (distance per level, weighted by size)."""
    n = cancel_d.size
    if n < 50:
        return {"ok": False, "n": int(n)}
    w_total = depth_w.sum()
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(BIN_LABELS):
        lo = BINS[i]
        hi = BINS[i + 1] - 1 if i + 1 < len(BINS) else np.inf
        c = ((cancel_d >= lo) & (cancel_d <= hi)).sum()
        dm = (depth_d >= lo) & (depth_d <= hi)
        d_share = float(depth_w[dm].sum() / w_total) if w_total > 0 else None
        c_share = float(c / n)
        rows.append(
            {
                "bucket": label,
                "cancel_share": c_share,
                "depth_share": d_share,
                "propensity": (c_share / d_share) if d_share and d_share > 0 else None,
            }
        )
    touch = rows[0]
    return {
        "ok": True,
        "n_cancels": int(n),
        "median_distance_ticks": float(np.median(cancel_d)),
        "at_touch_share": touch["cancel_share"],
        "at_touch_propensity": touch["propensity"],
        "buckets": rows,
    }


def lobster_cancel_gradient(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    book: dict[int, tuple[float, int]] = {}  # oid -> (price, direction)
    cancel_d: list[float] = []
    depth_d: list[float] = []
    depth_w: list[float] = []
    prev_top: tuple[float, float] | None = None  # (best_ask, best_bid)
    row_i = 0
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            if ev.event_type == SUBMISSION:
                book[ev.order_id] = (float(ev.price), ev.direction)
            elif ev.event_type in (DELETE, CANCEL_PARTIAL):
                rec = book.get(ev.order_id)
                if rec is not None and prev_top is not None:
                    px, side = rec
                    if side == 1:  # resting buy: distance below best bid
                        cancel_d.append(max(0.0, (prev_top[1] - px) / 100.0))
                    else:  # resting sell: distance above best ask
                        cancel_d.append(max(0.0, (px - prev_top[0]) / 100.0))
                if ev.event_type == DELETE:
                    book.pop(ev.order_id, None)
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                prev_top = (float(asks[0][0]), float(bids[0][0]))
                if row_i % 100 == 0:  # sample the depth histogram
                    b0 = float(bids[0][0])
                    a0 = float(asks[0][0])
                    for px, sz in bids:
                        depth_d.append((b0 - px) / 100.0)
                        depth_w.append(float(sz))
                    for px, sz in asks:
                        depth_d.append((px - a0) / 100.0)
                        depth_w.append(float(sz))
            row_i += 1
    return _gradient(np.asarray(cancel_d), np.asarray(depth_d), np.asarray(depth_w))


def sim_cancel_gradient(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    cancel_d: list[float] = []
    depth_d: list[float] = []
    depth_w: list[float] = []
    for i in range(horizon):
        before = dict(sim._orders)
        # Touch BEFORE the event: the cancel was placed against the
        # pre-event book (matches the real arm's prior-orderbook row).
        pre_bb, pre_ba = sim.best_bid_level, sim.best_ask_level
        ev = sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        if ev == "cancel" and pre_bb is not None and pre_ba is not None:
            for oid in before.keys() - set(sim._orders):
                o = before[oid]
                touch = pre_bb if o.side == "buy" else pre_ba
                cancel_d.append(float(abs(touch - o.level)))
        if i % 50 == 0 and bb is not None and ba is not None:
            for o in sim._orders.values():
                touch = bb if o.side == "buy" else ba
                depth_d.append(float(abs(touch - o.level)))
                depth_w.append(1.0)
    return _gradient(np.asarray(cancel_d), np.asarray(depth_d), np.asarray(depth_w))


def cancel_gradient_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_cancel_gradient(msg_path, ob_path)
    arms = {
        "iid": sim_cancel_gradient(seed=seed),
        "regime": sim_cancel_gradient(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_cancel_gradient(
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
    if real.get("ok") and real.get("at_touch_propensity") is not None:
        for name, arm in arms.items():
            if arm.get("ok") and arm.get("at_touch_propensity") is not None:
                ratio = arm["at_touch_propensity"] / real["at_touch_propensity"]
                if ratio < 0.5 or ratio > 2.0:
                    divergences.append(
                        f"{name}_touchprop_{arm['at_touch_propensity']:.3f}_vs_{real['at_touch_propensity']:.3f}"
                    )
    payload: dict[str, Any] = {
        "kind": "cancel_gradient",
        "schema": "cancel_gradient.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "cancel_propensity_gradient_measured",
        "interpretation": (
            "Cancel propensity = share of cancels at distance bucket d "
            "divided by share of resting volume at d. >1 at the touch = "
            "queue-skimming exits; ~1 uniform = inventory-driven exit. "
            "LOBSTER ids link DELETE to SUBMISSION price; distances use "
            "the pre-event touch. Sim orders carry level directly. "
            "Depth distribution is volume-weighted (per-share basis)."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
