"""Multi-level sweep width — the instantaneous aggressor footprint.

When one market order exhausts several resting orders, LOBSTER logs
each resting-order fill as a separate EXECUTION event sharing the same
timestamp. A same-timestamp, same-aggressor cluster of visible execs is
therefore one aggressor impulse; its footprint is the set of distinct
prices it prints (levels walked) and its total shares. Hidden execs
(type 5) are excluded — they cannot be attributed to the visible book.

Per aggressor cluster: ``n_fills``, ``levels`` (distinct prices),
``shares``, and ``walk_ticks`` (max price excursion from the first
print, in ticks — strictly monotone when the impulse climbs the
ladder). The width/levels distributions on the real tape vs the sim
arms measure a gap the current flow classes cannot express: ZI-LOB
emits unit-size aggressors, so every cluster is width 1.

Receipt kind ``sweep_width.v1``, data_label MIXED.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, parse_messages
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _footprint(clusters: list[tuple[int, list[tuple[float, float]]]]) -> dict[str, Any]:
    """clusters: (n_fills, [(price_ticks, size), ...])."""
    widths: list[int] = []  # distinct prices printed by the impulse
    fills: list[int] = []
    walks: list[float] = []
    shares: list[float] = []
    for _nf, prints in clusters:
        widths.append(len({p for p, _s in prints}))
        fills.append(len(prints))
        shares.append(sum(s for _p, s in prints))
        first = prints[0][0]
        walks.append(max(abs(p - first) for p, _s in prints))
    w = np.asarray(widths, dtype=float)
    return {
        "ok": len(clusters) >= 50,
        "n_clusters": len(clusters),
        "n_fills_mean": float(np.mean(fills)),
        "width_ge2_share": float(np.mean(w >= 2)),
        "width_ge4_share": float(np.mean(w >= 4)),
        "max_width": int(w.max()),
        "mean_walk_ticks": float(np.mean(walks)),
        "mean_shares": float(np.mean(shares)),
        "width_hist": {str(k): int(v) for k, v in sorted(Counter(widths).items())},
    }


def lobster_sweep_width(msg_path: Path) -> dict[str, Any]:
    """Group visible execs by (timestamp, aggressor sign)."""
    groups: dict[tuple[float, int], list[tuple[float, float]]] = {}
    for ev in parse_messages(msg_path):
        if ev.event_type == EXECUTION:
            key = (ev.time_s, -ev.direction)
            groups.setdefault(key, []).append((ev.price / 100.0, float(ev.size)))
    return _footprint([(len(v), v) for v in groups.values()])


def sim_sweep_width(
    flow: Any | None = None,
    *,
    seed: int = 0,
    horizon: int = 30000,
    tick: float = 0.01,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    groups: dict[tuple[float, str], list[tuple[float, float]]] = {}
    for _ in range(horizon):
        sim.step()
        for tr in sim.trades:
            key = (tr.t, tr.aggressor)
            groups.setdefault(key, []).append((tr.price / tick, float(tr.qty)))
        sim.trades.clear()
    return _footprint([(len(v), v) for v in groups.values()])


def sweep_width_bench(data_dir: Path) -> dict[str, Any]:
    msg = data_dir / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = data_dir / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg.exists() or not ob.exists():
        raise FileNotFoundError(f"LOBSTER tape required under {data_dir}")
    real = lobster_sweep_width(msg)
    arms = {
        "iid": sim_sweep_width(None),
        "regime": sim_sweep_width(
            MarkovRegimeFlow(
                states=(
                    RegimeState("calm", 1.0, 0.5),
                    RegimeState("bursty", 3.0, 0.62),
                ),
                stay_probs=(0.995, 0.985),
                seed=7,
            )
        ),
        "split": sim_sweep_width(SplitFlow(seed=11)),
    }
    divergences = [
        f"{name}_widthge2_{a['width_ge2_share']:.3f}_vs_{real['width_ge2_share']:.3f}"
        for name, a in arms.items()
        if abs(a["width_ge2_share"] - real["width_ge2_share"]) > 0.02
    ]
    # structural ceiling: real impulses walk >=4 levels while a unit-qty
    # arm can never print width > 1
    divergences += [
        f"{name}_inexpressible_maxwidth_{a['max_width']}_vs_{real['max_width']}"
        for name, a in arms.items()
        if real["max_width"] >= 4 and a["max_width"] < 2
    ]
    payload: dict[str, Any] = {
        "kind": "sweep_width",
        "schema": "sweep_width.v1",
        "tape": {"msg": msg.name, "ob": ob.name, "symbol": "AMZN", "date": "2012-06-21"},
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "aggressor_sweep_width_real_vs_sim",
        "interpretation": (
            "One aggressor impulse = same-timestamp, same-side cluster "
            "of visible EXECUTION rows (LOBSTER prints one row per "
            "resting order filled by a single MO). width = distinct "
            "prices printed; walk_ticks = price excursion of the "
            "cluster. Real tape sweeps walk ladders (width>=2 share, "
            "max width); every sim arm is pinned at width 1 because "
            "aggressor qty is unit-size — multi-level consumption is "
            "inexpressible in the current flow classes."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
