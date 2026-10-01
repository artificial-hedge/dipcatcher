"""State-duration check — Markov states must dwell geometrically.

A first-order Markov state with stay-probability ``A_ii`` has geometric
dwell times: ``P(dwell = k) = A_ii^{k-1} (1 - A_ii)``. If the decoded
state path shows materially non-geometric dwell (e.g. an enforced
minimum duration, a heavy dwell tail), the process is semi-Markov and
the HMM is misspecified — a real diagnostic, not a style nit.

Method: simulate a trajectory, Viterbi-decode it, collect dwell times
per state, and compare the empirical dwell survival curve against the
implied geometric one. The discrepancy is the max |ECDF - geometric|
(DKWM-style; we report the statistic, not a p-value — the bench pins
planted-geometric low and planted-fixed-duration high).

Sealed ``duration_check.v1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.hmm.discrete import DiscreteHMM, viterbi
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def _dwells(states: NDArray[np.int64], k: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = states.size
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        if states[i] == k:
            out.append(j - i)
        i = j
    return out


def dwell_discrepancy(model: DiscreteHMM, states: NDArray[np.int64]) -> dict[str, Any]:
    """Per-state max |ECDF(dwell) - Geom(1 - A_ii)| for a decoded path."""
    report: dict[str, Any] = {}
    for k in range(model.n_states):
        dwells = _dwells(states, k)
        if len(dwells) < 10:
            report[f"state_{k}"] = {"n_dwells": len(dwells), "status": "too_few"}
            continue
        a_ii = float(model.A[k, k])
        if a_ii >= 1.0:
            report[f"state_{k}"] = {"n_dwells": len(dwells), "status": "absorbing"}
            continue
        d = np.asarray(dwells)
        grid = np.arange(1, d.max() + 1)
        ecdf = np.array([np.mean(d <= g) for g in grid])
        geom = 1.0 - a_ii**grid  # P(dwell <= g) for geometric
        report[f"state_{k}"] = {
            "n_dwells": len(dwells),
            "a_ii": a_ii,
            "implied_mean_dwell": 1.0 / (1.0 - a_ii),
            "empirical_mean_dwell": float(d.mean()),
            "max_ecdf_gap": float(np.abs(ecdf - geom).max()),
            "status": "ok",
        }
    return report


def _simulate(
    model: DiscreteHMM, n: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    states = np.empty(n, dtype=np.int64)
    obs = np.empty(n, dtype=np.int64)
    states[0] = int(rng.choice(model.n_states, p=model.pi))
    obs[0] = int(rng.choice(model.n_obs, p=model.B[states[0]]))
    for t in range(1, n):
        states[t] = int(rng.choice(model.n_states, p=model.A[states[t - 1]]))
        obs[t] = int(rng.choice(model.n_obs, p=model.B[states[t]]))
    return obs, states


def duration_bench(n: int = 3000, seed: int = 0) -> dict[str, Any]:
    """Planted-geometric (should pass) vs planted-fixed-duration (should flag)."""
    rng = np.random.default_rng(seed)
    hmm = DiscreteHMM(
        A=np.array([[0.94, 0.06], [0.08, 0.92]]),
        B=np.array([[0.8, 0.2], [0.15, 0.85]]),
        pi=np.array([0.5, 0.5]),
    )
    obs, true_states = _simulate(hmm, n, rng)
    path, _ = viterbi(hmm, obs)
    report_true = dwell_discrepancy(hmm, true_states)
    report_dec = dwell_discrepancy(hmm, np.asarray(path))

    # planted semi-Markov: state 0 dwells exactly 20 bars each visit
    sm_states = np.empty(n, dtype=np.int64)
    t = 0
    while t < n:
        sm_states[t : min(t + 20, n)] = 0
        t += 20
        dwell = int(rng.geometric(0.05))
        sm_states[t : min(t + dwell, n)] = 1
        t += dwell
    # fit dwell report against an HMM pretending to model it
    sm_model = DiscreteHMM(
        A=np.array([[0.95, 0.05], [0.05, 0.95]]),
        B=np.array([[0.8, 0.2], [0.15, 0.85]]),
        pi=np.array([0.5, 0.5]),
    )
    report_sm = dwell_discrepancy(sm_model, sm_states)

    def _max_gap(rep: dict[str, Any]) -> float:
        return max(
            (v["max_ecdf_gap"] for v in rep.values() if v.get("status") == "ok"),
            default=0.0,
        )

    verdict = "ok" if _max_gap(report_true) < 0.15 and _max_gap(report_sm) > 0.25 else "weak"
    payload: dict[str, Any] = {
        "kind": "duration_check",
        "schema": "duration_check.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "geometric dwell stays geometric (gap small); fixed-duration states flag non-geometric (gap large)",
            "verdict": verdict,
            "max_gap_true_markov": _max_gap(report_true),
            "max_gap_decoded": _max_gap(report_dec),
            "max_gap_semimarkov": _max_gap(report_sm),
        },
        "interpretation": {
            "true_path": report_true,
            "decoded_path": report_dec,
            "planted_semimarkov": report_sm,
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
