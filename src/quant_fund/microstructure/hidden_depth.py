"""hidden_depth — hidden/iceberg liquidity share on the real tape.

LOBSTER emits EXECUTION_HIDDEN (type 5) for fills against hidden
liquidity — iceberg replenishments and fully hidden orders. The hidden
share of executed volume is a real-tape structural fact; the ZI-LOB sim
has no hidden orders at all, so the comparison is a pure mechanism gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, EXECUTION_HIDDEN, parse_messages
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _size_moments(sizes: list[int]) -> dict[str, float] | dict[str, Any]:
    if not sizes:
        return {"n": 0}
    arr = np.asarray(sizes, dtype=float)
    return {
        "n": int(arr.size),
        "mean_size": round(float(arr.mean()), 2),
        "median_size": round(float(np.median(arr)), 2),
        "p95_size": round(float(np.percentile(arr, 95)), 2),
    }


def lobster_hidden_share(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Split executions into visible vs hidden fills."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    vis_sizes: list[int] = []
    hid_sizes: list[int] = []
    vis_notional = hid_notional = 0
    for ev in parse_messages(msg):
        if ev.event_type == EXECUTION:
            vis_sizes.append(ev.size)
            vis_notional += ev.size * ev.price
        elif ev.event_type == EXECUTION_HIDDEN:
            hid_sizes.append(ev.size)
            hid_notional += ev.size * ev.price
    n = len(vis_sizes) + len(hid_sizes)
    if n == 0:
        return {"ok": False, "reason": "no_executions"}
    tot_notional = vis_notional + hid_notional
    return {
        "n_execs": n,
        "n_hidden_execs": len(hid_sizes),
        "hidden_trade_share": round(len(hid_sizes) / n, 4),
        "hidden_volume_share": round(sum(hid_sizes) / max(1, sum(vis_sizes) + sum(hid_sizes)), 4),
        "hidden_notional_share": round(hid_notional / max(1, tot_notional), 4),
        "visible": _size_moments(vis_sizes),
        "hidden": _size_moments(hid_sizes),
    }


def sim_hidden_share(horizon: int = 8000, seed: int = 7) -> dict[str, Any]:
    """The ZI-LOB sim has no hidden orders — honest zero."""
    sim = ZILobSimulator(ZILobConfig(seed=seed))
    for _ in range(horizon):
        sim.step()
    return {
        "n_execs": len(sim.trades),
        "hidden_trade_share": 0.0,
        "hidden_volume_share": 0.0,
        "mechanism_present": False,
    }


def hidden_depth_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Hidden liquidity share real vs sim. Sealed."""
    real = lobster_hidden_share(tape_dir, ticker)
    sim = sim_hidden_share(seed=seed)
    divergences = []
    if real.get("hidden_trade_share") is not None and real["hidden_trade_share"] > 0.02:
        divergences.append("sim_has_no_hidden_liquidity_mechanism")
    payload: dict[str, Any] = {
        "kind": "hidden_depth",
        "schema": "hidden_depth.v1",
        "ticker": ticker,
        "real": real,
        "sim": sim,
        "divergences": divergences,
        "claim": "hidden_liquidity_share_measured",
        "interpretation": (
            "EXECUTION_HIDDEN counts iceberg/hidden fills. A sim without "
            "hidden orders cannot model the iceberg retreat-and-reload "
            "dynamic that makes displayed depth understate true depth."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
