"""lob_resilience — how fast does a depleted level refill?

When a market order sweeps (or cancels wipe) the visible size at the
touch, the time until that level's displayed size recovers is the
market's *resiliency* — a canonical LOB property (Kempf–Mayston 2008;
Large 2007). Real books refill in tens of ms; a book where liquidity
isn't posted back is not resilient.

Real: track the touch price's displayed size; a depletion event is
size dropping ≥80% in one event; refill = first subsequent book row
where the size at that same price is ≥50% of pre-depletion. The sim
tracks the analogous touch level.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

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

DEPLETION = 0.2  # size drops to ≤20% of prior
RECOVERY = 0.5  # recovered when ≥50% of prior returns


def _resilience_stats(
    times: list[float],
    sizes: list[int],
) -> dict[str, Any]:
    """Depletion→refill stats on a scalar touch-size stream."""
    refill_s: list[float] = []
    n_depletions = 0
    n_unrecovered = 0
    i = 0
    n = len(sizes)
    while i + 1 < n:
        s0 = sizes[i]
        s1 = sizes[i + 1]
        if s0 > 0 and s1 <= DEPLETION * s0:
            n_depletions += 1
            found = False
            for j in range(i + 1, n):
                if sizes[j] >= RECOVERY * s0:
                    refill_s.append(times[j] - times[i + 1])  # from depletion event
                    i = j - 1
                    found = True
                    break
            if not found:
                # An unrecovered depletion ends ITS episode, not the scan:
                # later drops vs later baselines are still depletions.
                n_unrecovered += 1
        i += 1
    arr = np.asarray(refill_s)
    out: dict[str, Any] = {
        "n_depletions": n_depletions,
        "n_recovered": len(refill_s),
        "n_unrecovered": n_unrecovered,
        "recovery_share": round(len(refill_s) / n_depletions, 4) if n_depletions else 0.0,
    }
    if arr.size:
        out["median_refill_s"] = round(float(np.median(arr)), 4)
        out["p90_refill_s"] = round(float(np.percentile(arr, 90)), 4)
        out["mean_refill_s"] = round(float(arr.mean()), 4)
    return out


def lobster_resilience(tape_dir: Path, ticker: str = "AMZN", side: str = "ask") -> dict[str, Any]:
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    times: list[float] = []
    sizes: list[int] = []
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    with ob.open() as fo:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(fo), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            touch = asks[0] if (side == "ask" and asks) else (bids[0] if bids else None)
            if touch:
                sizes.append(int(touch[1]))
                times.append(ev.time_s)
    return _resilience_stats(times, sizes)


def sim_resilience(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    rng: Generator | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
    side: str = "ask",
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    times: list[float] = []
    sizes: list[int] = []
    for _ in range(horizon):
        sim.step()
        lvl = sim.best_ask_level if side == "ask" else sim.best_bid_level
        if lvl is not None:
            sizes.append(sim.depth_at("sell" if side == "ask" else "buy", lvl))
            times.append(sim.t)
    return _resilience_stats(times, sizes)


def resilience_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Real vs sim refill dynamics. Sealed."""
    real = {
        "ask": lobster_resilience(tape_dir, ticker, "ask"),
        "bid": lobster_resilience(tape_dir, ticker, "bid"),
    }
    arms = {
        "iid": sim_resilience(seed=seed),
        "regime": sim_resilience(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_resilience(
            flow=SplitFlow(
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
    divergences = []
    r_med = real["ask"].get("median_refill_s")
    for name, arm in arms.items():
        a_med = arm.get("median_refill_s")
        if r_med and a_med and abs(a_med - r_med) / r_med > 0.8:
            divergences.append(f"{name}_refill_ratio_{a_med / r_med:.2f}")
    payload: dict[str, Any] = {
        "kind": "lob_resilience",
        "schema": "lob_resilience.v1",
        "ticker": ticker,
        "depletion_rule": f"size_drops_to_<={DEPLETION}x_prior",
        "recovery_rule": f"size_recovers_to_>={RECOVERY}x_prior",
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "resilience_measured",
        "interpretation": (
            "Depleted touch levels that never refill indicate "
            "liquidity withdrawal; fast refill = resilient book."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
