"""Anytime-valid policy dominance — paired ZI-LOB episodes + bounded-mean
e-process (ULTRAPLAN lane; pairs with ``mean_eprocess`` / ``loss_cs``).

Two market-making quote policies are evaluated on **common random numbers**:
episode ``t`` runs policy A and policy B on fresh simulators seeded by the
same episode seed (identical arrival streams), so the paired difference
``d_t = pnl_B - pnl_A`` is a clean counterfactual contrast — not the mean of
two independent runs.

``d_t`` is generally unbounded; to use the bounded-mean e-process we declare
a contract ``|d_t| <= pnl_scale`` (fail closed on violation — never clipped,
which would bias toward the bound) and map to the unit interval via
``x_t = 0.5 + d_t / (2 * pnl_scale)``. Then ``E[x] > 0.5`` iff the
challenger's mean episode score exceeds the baseline's, and
``MeanEProcess`` yields a time-uniform lower bound on ``E[x]`` plus an
anytime-valid dominance alarm at ``E[x] > 0.5``.

Honesty: ``pnl`` here is the *simulator-internal* mark-to-market diagnostic
(``sim_internal_mtm_pnl_final``) of the synthetic ZI-LOB engine — the
receipt records it only under ``sim_internal_*`` names, marks the evidence
``SYNTHETIC``, and makes no live-trading or market claim.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    QuotePolicy,
    RegimeState,
    ZILobConfig,
    run_mm_session,
    santa_fe_config,
)
from quant_fund.research.mean_eprocess import MeanEProcess
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

POLICY_EPROCESS_SCHEMA = "policy_eprocess.v1"

DEFAULT_TWO_STATE_FLOW: tuple[RegimeState, RegimeState] = (
    RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
    RegimeState(name="stress", intensity_mult=2.5, p_buy=0.7),
)


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _episode_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"episode seed must be a non-negative int, got {seed!r}")
    return seed


def dominance_stream(
    pnl_baseline: Sequence[float] | NDArray[np.float64],
    pnl_challenger: Sequence[float] | NDArray[np.float64],
    *,
    pnl_scale: float,
) -> NDArray[np.float64]:
    """Paired bounded dominance stream ``x_t = 0.5 + (b_t - a_t)/(2*scale)``.

    ``E[x] > 0.5`` iff ``E[b] > E[a]``; ``E[x] < 0.5`` iff the baseline
    dominates. Both inputs must be finite and equally long; any
    ``|b_t - a_t| > pnl_scale`` raises — the declared bound is a contract,
    so out-of-contract observations fail closed rather than being clipped.
    """
    scale = _pos_finite(pnl_scale, "pnl_scale")
    a = np.asarray(pnl_baseline, dtype=float)
    b = np.asarray(pnl_challenger, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("pnl streams must be 1-D arrays of equal length")
    if a.size == 0:
        raise ValueError("pnl streams must be non-empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("pnl streams must be finite")
    diff = b - a
    if np.any(np.abs(diff) > scale):
        raise ValueError(
            f"pnl_scale contract violated: max|d|={float(np.abs(diff).max()):.6g} > {scale:.6g}"
        )
    return np.asarray(0.5 + diff / (2.0 * scale), dtype=np.float64)


@dataclass(frozen=True)
class PairedEpisode:
    """One paired episode: same episode seed for both policies."""

    seed: int
    pnl_a: float
    pnl_b: float
    n_fills_a: int
    n_fills_b: int


def run_paired_episode(
    config: ZILobConfig,
    policy_a: QuotePolicy,
    policy_b: QuotePolicy,
    *,
    horizon: float,
    episode_seed: int,
    flow_states: tuple[RegimeState, RegimeState] = DEFAULT_TWO_STATE_FLOW,
    stay_probs: tuple[float, float] = (0.98, 0.98),
    decision_interval: float = 1.0,
    inventory_cap: int | None = None,
) -> PairedEpisode:
    """Run both policies on identical streams (common random numbers)."""
    seed = _episode_seed(episode_seed)
    _pos_finite(horizon, "horizon")

    def _run(policy: QuotePolicy) -> dict[str, Any]:
        flow = MarkovRegimeFlow(flow_states, stay_probs, seed=seed)
        cfg = dataclasses.replace(config, seed=seed)
        return run_mm_session(
            config=cfg,
            policy=policy,
            horizon=horizon,
            decision_interval=decision_interval,
            flow=flow,
            inventory_cap=inventory_cap,
        )

    ra, rb = _run(policy_a), _run(policy_b)
    return PairedEpisode(
        seed=seed,
        pnl_a=float(ra["sim_internal_mtm_pnl_final"]),
        pnl_b=float(rb["sim_internal_mtm_pnl_final"]),
        n_fills_a=int(ra["n_fills"]),
        n_fills_b=int(rb["n_fills"]),
    )


def policy_dominance_process(
    policy_a: QuotePolicy,
    policy_b: QuotePolicy,
    *,
    config: ZILobConfig,
    horizon: float,
    n_episodes: int,
    pnl_scale: float,
    seed: int = 0,
    decision_interval: float = 1.0,
    inventory_cap: int | None = None,
) -> dict[str, Any]:
    """Stream paired episodes into ``MeanEProcess`` on the dominance share.

    Returns the process (with its final bounds/alarm) plus the episode rows.
    ``E[x] > 0.5`` means the challenger (policy_b) dominates.
    """
    _pos_finite(horizon, "horizon")
    _pos_finite(pnl_scale, "pnl_scale")
    if isinstance(n_episodes, bool) or int(n_episodes) < 1:
        raise ValueError(f"n_episodes must be an int >= 1, got {n_episodes!r}")
    proc = MeanEProcess()
    episodes: list[PairedEpisode] = []
    xs: list[float] = []
    for k in range(int(n_episodes)):
        ep = run_paired_episode(
            config,
            policy_a,
            policy_b,
            horizon=horizon,
            episode_seed=seed + k,
            decision_interval=decision_interval,
            inventory_cap=inventory_cap,
        )
        episodes.append(ep)
        x = float(dominance_stream([ep.pnl_a], [ep.pnl_b], pnl_scale=pnl_scale)[0])
        xs.append(x)
        proc.update(x)
    return {"process": proc, "episodes": episodes, "x": np.asarray(xs)}


def policy_eprocess_bench(
    policy_a: QuotePolicy,
    policy_b: QuotePolicy,
    *,
    name_a: str,
    name_b: str,
    n_episodes: int = 12,
    horizon: float = 400.0,
    pnl_scale: float = 2.0,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Sealed ``policy_eprocess.v1`` receipt for an A-vs-B dominance run."""
    _pos_finite(pnl_scale, "pnl_scale")
    if not name_a or not name_b or name_a == name_b:
        raise ValueError("distinct nonempty policy names required")
    out = policy_dominance_process(
        policy_a,
        policy_b,
        config=santa_fe_config(seed=999),
        horizon=horizon,
        n_episodes=n_episodes,
        pnl_scale=pnl_scale,
        seed=seed,
        inventory_cap=5,
    )
    proc: MeanEProcess = out["process"]
    episodes: list[PairedEpisode] = out["episodes"]
    x: NDArray[np.float64] = out["x"]
    lo, hi = proc.interval(alpha)
    verdict = "b_dominates" if lo > 0.5 else ("a_dominates" if hi < 0.5 else "inconclusive")
    payload: dict[str, Any] = {
        "schema": POLICY_EPROCESS_SCHEMA,
        "kind": "policy_eprocess",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Paired synthetic-ZI-LOB dominance e-process; episode scores are "
            "sim_internal diagnostics, never market evidence."
        ),
        "policies": {"a": name_a, "b": name_b},
        "n_episodes": int(n_episodes),
        "horizon": float(horizon),
        "pnl_scale": float(pnl_scale),
        "alpha": float(alpha),
        "mean_share": float(x.mean()),
        "cs_lower": float(lo),
        "cs_upper": float(hi),
        "alarm_b_dominates": bool(proc.alarm_above(0.5)),
        "alarm_a_dominates": bool(proc.alarm_below(0.5)),
        "verdict": verdict,
        "sim_internal_pnl_a": [e.pnl_a for e in episodes],
        "sim_internal_pnl_b": [e.pnl_b for e in episodes],
        "n_fills_a": [e.n_fills_a for e in episodes],
        "n_fills_b": [e.n_fills_b for e in episodes],
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "DEFAULT_TWO_STATE_FLOW",
    "POLICY_EPROCESS_SCHEMA",
    "PairedEpisode",
    "dominance_stream",
    "policy_dominance_process",
    "policy_eprocess_bench",
    "run_paired_episode",
]
