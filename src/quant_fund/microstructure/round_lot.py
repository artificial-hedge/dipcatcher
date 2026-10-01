"""round_lot — trade-size distribution: the round-lot preference.

Real market participants submit orders in round lots — 100s, 200s,
1000s — not lognormal draws. The exec-size distribution is lumpy with
spikes at round numbers; the size histogram + round-share is a cheap,
sharp distributional fact the sim's size generator either has or lacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, parse_messages
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

ROUND_LOTS = (50, 100, 200, 300, 400, 500, 1000, 2000, 5000)
SIZE_BINS = (0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 2500, 1 << 30)


def _size_profile(sizes: np.ndarray) -> dict[str, Any]:
    if sizes.size == 0:
        return {"ok": False, "reason": "no_trades"}
    hist = {
        f"{lo}-{hi if hi < (1 << 30) else 'inf'}": int(((sizes >= lo) & (sizes < hi)).sum())
        for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:], strict=True)
    }
    round_share = float(np.isin(sizes, ROUND_LOTS).mean())
    hundred_share = float((sizes % 100 == 0).mean())
    return {
        "n_trades": int(sizes.size),
        "mean_size": round(float(sizes.mean()), 2),
        "median_size": round(float(np.median(sizes)), 2),
        "round_lot_share": round(round_share, 4),
        "multiple_of_100_share": round(hundred_share, 4),
        "size_hist": hist,
        "p95_size": round(float(np.percentile(sizes, 95)), 2),
    }


def lobster_round_lot(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    sizes = np.asarray(
        [ev.size for ev in parse_messages(msg) if ev.event_type == EXECUTION],
        dtype=float,
    )
    out = _size_profile(sizes)
    out["mode"] = "execution_sizes"
    return out


def sim_round_lot(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(horizon):
        sim.step()
    sizes = np.asarray([tr.qty for tr in sim.trades], dtype=float)
    return _size_profile(sizes)


def round_lot_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Size-distribution comparison real vs sim. Sealed."""
    real = lobster_round_lot(tape_dir, ticker)
    arms = {
        "iid": sim_round_lot(seed=seed),
        "regime": sim_round_lot(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_round_lot(
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
    for name, arm in arms.items():
        if arm.get("round_lot_share") is not None and real.get("round_lot_share") is not None:
            gap = real["round_lot_share"] - arm["round_lot_share"]
            if abs(gap) > 0.15:
                divergences.append(f"{name}_round_lot_share_gap_{gap:+.2f}")
    payload: dict[str, Any] = {
        "kind": "round_lot",
        "schema": "round_lot.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "round_lot_size_distribution_measured",
        "interpretation": (
            "Real tape lumps at round lots; a smooth size generator "
            "understates round_share. Gap > 0.15 flags the divergence."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
