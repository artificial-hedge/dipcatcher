"""streak_stats — same-sign execution run lengths on real tape vs sim.

Metaorder slicing produces runs of consecutive same-sign fills. The
run-length histogram is the direct observable: under iid signs the
distribution is geometric (P(run=k) = 2^{-k}); real tape shows a fat
right tail — long slicing runs. Excess mass at k>5 marks correlated
flow the sim's regime/split mechanisms must reproduce.
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


def _run_lengths(signs: list[int]) -> np.ndarray:
    if not signs:
        return np.asarray([], dtype=float)
    runs: list[int] = []
    cur = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.asarray(runs, dtype=float)


def _streak_stats(runs: np.ndarray) -> dict[str, Any]:
    if runs.size == 0:
        return {"ok": False, "reason": "no_runs"}
    max_k = 20
    hist = np.zeros(max_k, dtype=float)
    for r in runs:
        hist[min(int(r), max_k) - 1] += 1
    emp = hist / hist.sum()
    geo = np.asarray([0.5**k for k in range(1, max_k + 1)])
    geo /= geo.sum()
    excess_mass_gt5 = float(emp[5:].sum() - geo[5:].sum())
    return {
        "n_runs": int(runs.size),
        "mean_run": round(float(runs.mean()), 3),
        "max_run": int(runs.max()),
        "p_ge_5": round(float((runs >= 5).mean()), 4),
        "p_ge_10": round(float((runs >= 10).mean()), 4),
        "excess_mass_gt5_vs_geo": round(excess_mass_gt5, 4),
        "run_hist": {str(k + 1): round(float(emp[k]), 4) for k in range(max_k)},
    }


def lobster_streaks(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    signs = [-ev.direction for ev in parse_messages(msg) if ev.event_type == EXECUTION]
    out = _streak_stats(_run_lengths(signs))
    out["n_execs"] = len(signs)
    return out


def sim_streaks(
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
    signs = [1 if tr.aggressor == "buy" else -1 for tr in sim.trades]
    out = _streak_stats(_run_lengths(signs))
    out["n_execs"] = len(signs)
    return out


def streak_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Run-length comparison real vs sim arms. Sealed."""
    real = lobster_streaks(tape_dir, ticker)
    arms = {
        "iid": sim_streaks(seed=seed),
        "regime": sim_streaks(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_streaks(
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
    real_xs = real.get("excess_mass_gt5_vs_geo")
    for name, arm in arms.items():
        xs = arm.get("excess_mass_gt5_vs_geo")
        if real_xs is not None and xs is not None and abs(real_xs - xs) > 0.05:
            divergences.append(f"{name}_run_tail_gap_{real_xs - xs:+.3f}")
    payload: dict[str, Any] = {
        "kind": "streak_stats",
        "schema": "streak_stats.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "exec_sign_run_length_measured",
        "interpretation": (
            "Run-length histogram is the discrete window on metaorder "
            "slicing: geometric under iid signs, fat-tailed under "
            "splitting. Excess mass beyond k=5 is the direct count."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
