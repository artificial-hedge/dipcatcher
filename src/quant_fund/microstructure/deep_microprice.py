"""Multi-level (deep) microprice vs the touch-only Stoikov microprice.

The touch microprice uses only level-1 imbalance. The depth-weighted
extension pulls k levels of the book:

    M_k = sum_j<=k (b_sz_j * a_px_j + a_sz_j * b_px_j) / sum_j<=k (b_sz_j + a_sz_j)

— each side's resting volume pulls the *opposite* side's price, so M_k is
a mid weighted toward the heavier side. The question the bench answers:
which depth k best predicts the next mid move? If deeper levels carry
signal, R^2 rises with k; if the touch is sufficient, k=1 dominates and
deeper depth is noise.

Layers
------
- ``_microprice_curves`` — M_k series + per-k OLS R^2 on forward Δmid.
- ``lobster_deep_microprice`` — real tape from the orderbook CSV.
- ``sim_deep_microprice`` — ZI-LOB from ``best_*_level`` + level sizes.
- ``deep_microprice_bench`` — divergence flags + sealed receipt.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import parse_messages, parse_orderbook_row
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

MAX_DEPTH = 5
FWD_EVENTS = 10


def _microprice_curves(
    asks: list[list[tuple[float, float]]], bids: list[list[tuple[float, float]]]
) -> dict[str, Any]:
    """asks/bids: per-event top-of-book ladders [(price, size), ...].

    Returns per-k R^2 of OLS forward-Δmid on M_k − mid."""
    mids: list[float] = []
    deep: list[list[float]] = []
    for a_ladder, b_ladder in zip(asks, bids, strict=True):
        if not a_ladder or not b_ladder:
            continue
        mids.append((a_ladder[0][0] + b_ladder[0][0]) / 2.0)
        vals: list[float] = []
        for k in range(1, MAX_DEPTH + 1):
            a = a_ladder[:k]
            b = b_ladder[:k]
            num = 0.0
            den = 0.0
            for j in range(min(len(a), len(b))):
                apx, asz = a[j]
                bpx, bsz = b[j]
                num += bsz * apx + asz * bpx
                den += bsz + asz
            vals.append(num / den if den > 0 else np.nan)
        deep.append(vals)
    mid = np.asarray(mids)
    dmp = np.asarray(deep)  # (n, MAX_DEPTH)
    n = mid.size
    if n < 200:
        return {"ok": False, "n": int(n)}
    dm = np.full(n, np.nan)
    dm[:-FWD_EVENTS] = mid[FWD_EVENTS:] - mid[:-FWD_EVENTS]
    valid = np.isfinite(dm) & np.isfinite(dmp).all(axis=1)
    per_k: list[dict[str, Any]] = []
    for k in range(MAX_DEPTH):
        x = (dmp[:, k] - mid)[valid]
        dv = dm[valid]
        if x.size < 50 or np.var(x) <= 0 or np.var(dv) <= 0:
            per_k.append({"levels": k + 1, "ok": False})
            continue
        xc = x - x.mean()
        dc = dv - dv.mean()
        slope = float(np.dot(xc, dc) / np.dot(xc, xc))
        resid = dc - slope * xc
        r2 = 1.0 - float(resid @ resid) / float(dc @ dc)
        r10 = np.corrcoef(x, dv)[0, 1]
        per_k.append(
            {
                "levels": k + 1,
                "ok": True,
                "slope": slope,
                "r2": r2,
                "corr": float(r10),
            }
        )
    best = max((p for p in per_k if p.get("ok")), key=lambda p: p["r2"], default=None)
    return {
        "ok": True,
        "n": int(n),
        "fwd_events": FWD_EVENTS,
        "per_depth": per_k,
        "best_depth": best["levels"] if best else None,
        "best_r2": best["r2"] if best else None,
    }


def lobster_deep_microprice(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    asks_all: list[list[tuple[float, float]]] = []
    bids_all: list[list[tuple[float, float]]] = []
    with ob_path.open() as f_ob:
        for _ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            asks_all.append([(p / 100.0, float(s)) for p, s in asks[:MAX_DEPTH]])
            bids_all.append([(p / 100.0, float(s)) for p, s in bids[:MAX_DEPTH]])
    return _microprice_curves(asks_all, bids_all)


def sim_deep_microprice(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    asks_all: list[list[tuple[float, float]]] = []
    bids_all: list[list[tuple[float, float]]] = []
    for _ in range(horizon):
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        a_ladder: list[tuple[float, float]] = []
        b_ladder: list[tuple[float, float]] = []
        if bb is not None and ba is not None:
            for j in range(MAX_DEPTH):
                sz_a = sim.depth_at("sell", ba + j)
                if sz_a:
                    a_ladder.append((sim.level_to_price(ba + j), float(sz_a)))
                sz_b = sim.depth_at("buy", bb - j)
                if sz_b:
                    b_ladder.append((sim.level_to_price(bb - j), float(sz_b)))
        asks_all.append(a_ladder)
        bids_all.append(b_ladder)
    return _microprice_curves(asks_all, bids_all)


def deep_microprice_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_deep_microprice(msg_path, ob_path)
    arms = {
        "iid": sim_deep_microprice(seed=seed),
        "regime": sim_deep_microprice(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_deep_microprice(
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
    if real.get("ok") and real.get("best_r2") is not None:
        for name, arm in arms.items():
            if (
                arm.get("ok")
                and arm.get("best_r2") is not None
                and arm["best_depth"] != real["best_depth"]
            ):
                divergences.append(f"{name}_bestdepth_{arm['best_depth']}_vs_{real['best_depth']}")
    payload: dict[str, Any] = {
        "kind": "deep_microprice",
        "schema": "deep_microprice.v1",
        "ticker": ticker,
        "max_depth": MAX_DEPTH,
        "fwd_events": FWD_EVENTS,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "depth_weighted_microprice_predictive_rank_measured",
        "interpretation": (
            "M_k = sum_j<=k (b_sz*a_px + a_sz*b_px)/sum_j<=k (b_sz+a_sz) "
            "— the Stoikov imbalance generalized to k levels. OLS R^2 of "
            "forward 10-event Δmid on M_k − mid per depth says which "
            "depth carries signal. best_depth/best_r2 give the argmax. "
            "Prices in ticks (LOBSTER price units / 100)."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
