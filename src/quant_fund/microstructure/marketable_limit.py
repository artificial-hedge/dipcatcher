"""marketable_limit — limit orders priced through the spread.

A buy submission priced ≥ best ask (sell ≤ best bid) is a marketable
limit order: taker aggression wearing a maker message. The share of
submissions that are marketable / at-touch / inside-spread / behind
is the placement-aggression spectrum of the tape.

The ZI-LOB sim places at distance ≥1 tick off the touch — marketable
submissions are structurally absent. Measured as a hard divergence.

Pre-event touch = previous orderbook row's top (ground truth).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import SUBMISSION, parse_messages, parse_orderbook_row
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


def _aggression_stats(rel: np.ndarray, rel_own: np.ndarray, sizes: np.ndarray) -> dict[str, Any]:
    """rel = (submit price − opposite touch) in ticks (>0 crosses);
    rel_own = (submit price − own touch) in ticks, sign-adjusted so
    >0 means strictly inside the spread, ==0 at own touch, <0 behind."""
    n = rel.size
    if n < 50:
        return {"ok": False, "n": int(n)}
    cats = {
        "marketable": rel >= 0,
        "crossing": rel > 0,
        "inside": (rel < 0) & (rel_own > 0),  # strictly inside the spread
        "at_own_touch": rel_own == 0,
        "behind": rel_own < 0,
    }
    shares = {k: float(v.mean()) for k, v in cats.items()}
    vol_shares = {
        k: float(sizes[v].sum() / sizes.sum()) for k, v in cats.items() if sizes.sum() > 0
    }
    return {
        "ok": True,
        "n": int(n),
        "share": shares,
        "volume_share": vol_shares,
        "mean_size_marketable": (
            float(sizes[cats["marketable"]].mean()) if cats["marketable"].any() else None
        ),
        "mean_size_resting": (float(sizes[rel <= 0].mean()) if (rel <= 0).any() else None),
        "rel_median_ticks": float(np.median(rel)),
    }


def lobster_marketable_limit(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    rels: list[float] = []
    rels_own: list[float] = []
    sizes: list[int] = []
    prev_top: tuple[int, int] | None = None  # (best_ask, best_bid)
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            if ev.event_type == SUBMISSION and ev.price > 0 and prev_top is not None:
                best_ask, best_bid = prev_top
                if ev.direction == 1:  # buy submit
                    rels.append((ev.price - best_ask) / 100.0)
                    rels_own.append((ev.price - best_bid) / 100.0)
                else:
                    rels.append((best_bid - ev.price) / 100.0)
                    rels_own.append((best_ask - ev.price) / 100.0)
                sizes.append(ev.size)
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                prev_top = (asks[0][0], bids[0][0])
    return _aggression_stats(np.asarray(rels), np.asarray(rels_own), np.asarray(sizes, dtype=float))


def sim_marketable_limit(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    rels: list[float] = []
    rels_own: list[float] = []
    sizes: list[int] = []
    for _ in range(horizon):
        before = set(sim._orders)
        ev = sim.step()
        if ev == "limit":
            for oid in set(sim._orders) - before:
                o = sim._orders[oid]
                if o.side == "buy" and sim.best_ask_level is not None:
                    rels.append(float(o.level - sim.best_ask_level))
                    rels_own.append(
                        float(o.level - sim.best_bid_level)
                        if sim.best_bid_level is not None
                        else float("nan")
                    )
                    sizes.append(1)
                elif o.side == "sell" and sim.best_bid_level is not None:
                    rels.append(float(sim.best_bid_level - o.level))
                    rels_own.append(
                        float(sim.best_ask_level - o.level)
                        if sim.best_ask_level is not None
                        else float("nan")
                    )
                    sizes.append(1)
    rel_own = np.asarray(rels_own)
    keep = ~np.isnan(rel_own)
    return _aggression_stats(
        np.asarray(rels)[keep], rel_own[keep], np.asarray(sizes, dtype=float)[keep]
    )


def marketable_limit_bench(
    tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7
) -> dict[str, Any]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook CSV pair under {tape_dir}")
    real = lobster_marketable_limit(msg[0], ob[0])
    arms = {
        "iid": sim_marketable_limit(seed=seed),
        "regime": sim_marketable_limit(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_marketable_limit(
            flow=SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        for key in ("marketable", "inside"):
            r = real["share"].get(key)
            if r is not None:
                for name, arm in arms.items():
                    if not arm.get("ok"):
                        continue
                    s = arm["share"].get(key)
                    if s is not None and abs(r - s) > 0.05:
                        divergences.append(f"{name}_{key}_{s:.3f}_vs_{r:.3f}")
    payload: dict[str, Any] = {
        "kind": "marketable_limit",
        "schema": "marketable_limit.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "placement_aggression_spectrum_measured",
        "interpretation": (
            "Share of SUBMISSIONs priced relative to the pre-event "
            "opposite and own touches: rel>=0 marketable (executes "
            "immediately: at or through the touch), rel>0 strictly "
            "crossing, inside = strictly inside the spread, "
            "at_own_touch = joins the own-side queue, behind = deeper. "
            "In LOBSTER data, marketable orders are logged as executions "
            "rather than resting submissions — a near-zero marketable "
            "share is a recording convention, not absent aggression. "
            "volume_share weights by size. The sim's placements are "
            "dist>=1 — marketable LOs structurally absent, an "
            "expressivity gap not a calibration miss."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
