"""HMM refit stability — is the learned regime decomposition stable under resampling?

A single Baum–Welch fit says nothing about whether the recovered states are
the *same* states under a refit: EM is non-convex and state labels are
unidentifiable up to permutation (label switching). This lane measures it:

1. fit a reference model on the event stream,
2. refit under seeded restarts on moving-block bootstrap resamples,
3. match each refit's states back to the reference by Hungarian assignment
   on the transition + emission matrices — a label-permuted but otherwise
   identical model scores zero deviation, so label switching never counts
   as instability,
4. report per-state assignment consistency, occupancy-share spread, and
   A/B Frobenius deviation on the *matched* labels.

``hmm_stability_bench`` plants a 2-state stream, checks that the fit
recovers the same states across restarts, and seals a ``hmm_stability.v1``
receipt. Research only; planted streams are SYNTHETIC correctness checks,
never market evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from quant_fund.hmm.discrete import DiscreteHMM, baum_welch, log_viterbi
from quant_fund.utils.atomicio import publish_text_once
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]

HMM_STABILITY_SCHEMA = "hmm_stability.v1"

#: fit(data, seed) -> fitted DiscreteHMM; seed drives any stochastic init.
FitFn = Callable[[IntArray, int], DiscreteHMM]

#: Default fraction of restarts that must stay under the deviation threshold.
_STABLE_SHARE = 0.9


@dataclass
class RefitStability:
    """Seeded bootstrap refits matched back to a reference fit's labels."""

    reference: DiscreteHMM
    seed: int
    n_restarts: int
    block_len: int
    stream_sha256: str
    # Per-restart Hungarian assignment ref state -> fitted state index.
    assignments: list[list[int]]
    # Refits re-labeled into reference coordinates, same order as assignments.
    matched: list[DiscreteHMM]
    failed: int


def _block_bootstrap(stream: IntArray, rng: np.random.Generator, block_len: int) -> IntArray:
    """Moving-blocks resample: circular blocks preserve serial structure."""
    n = stream.size
    n_blocks = max(1, math.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_len)[None, :]) % n
    return np.asarray(stream[idx.reshape(-1)[:n]], dtype=np.intp)


def _a_row_signature(A: Array) -> Array:
    """Permutation-invariant signature of each transition row.

    A relabeled model has ``A' = P A P.T``: row ``i`` of ``A'`` is a
    permuted copy of row ``p(i)`` of ``A``, so element-wise row comparison
    cannot detect label switching (the columns live in different label
    spaces). The self-transition mass plus the *sorted* off-diagonal masses
    is invariant to simultaneous row/column relabeling — each state's
    signature travels with it.
    """
    n = A.shape[0]
    off = np.sort(A[~np.eye(n, dtype=bool)].reshape(n, n - 1), axis=1)
    return np.concatenate([np.diag(A)[:, None], off], axis=1)


def _match_states(reference: DiscreteHMM, fitted: DiscreteHMM) -> IntArray:
    """Hungarian assignment of reference states to fitted states.

    ``cost[i, j] = ||sig(A_ref)[i] - sig(A_fit)[j]||^2 +
    ||B_ref[i] - B_fit[j]||^2`` — a permutation-invariant transition-row
    signature plus the emission row, so a label-permuted but otherwise
    identical fit scores zero and label switching never counts as
    instability.
    """
    if fitted.n_states != reference.n_states:
        raise ValueError(f"refit has {fitted.n_states} states; reference has {reference.n_states}")
    if fitted.n_obs != reference.n_obs:
        raise ValueError(
            f"refit has {fitted.n_obs} observation symbols; reference has {reference.n_obs}"
        )
    sig_ref = _a_row_signature(reference.A)
    sig_fit = _a_row_signature(fitted.A)
    cost = ((sig_ref[:, None, :] - sig_fit[None, :, :]) ** 2).sum(axis=2) + (
        (reference.B[:, None, :] - fitted.B[None, :, :]) ** 2
    ).sum(axis=2)
    _, assigned = linear_sum_assignment(cost)
    return np.asarray(assigned, dtype=np.intp)


def _relabel(model: DiscreteHMM, assignment: IntArray) -> DiscreteHMM:
    """Re-express *model* in reference coordinates: state i := fitted assignment[i]."""
    return DiscreteHMM(
        model.A[assignment][:, assignment],
        model.B[assignment],
        model.pi[assignment],
    )


def _canonical_order(model: DiscreteHMM) -> IntArray:
    """Deterministic canonical state order, independent of EM's label choice.

    Sort key is the state's emission row, tie-broken by its transition-row
    signature — so the same decomposition receives the same labels no matter
    which label permutation a fit converged to. Applied to the reference and
    every refit before matching, assignment indices become comparable across
    restarts and ``assignment_consistency`` is meaningful.
    """
    sig = _a_row_signature(model.A)
    keys = [tuple(model.B[s].tolist()) + tuple(sig[s].tolist()) for s in range(model.n_states)]
    return np.asarray(sorted(range(model.n_states), key=keys.__getitem__), dtype=np.intp)


def _stationary_share(A: Array) -> Array:
    """Stationary distribution of a row-stochastic transition matrix.

    Least-squares solve of ``(A.T - I) pi = 0`` subject to ``sum(pi) = 1``;
    degenerate chains fall back to the uniform share.
    """
    n = A.shape[0]
    system = np.concatenate([A.T - np.eye(n), np.ones((1, n))])
    rhs = np.concatenate([np.zeros(n), np.ones(1)])
    solved, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    share = np.clip(np.asarray(solved, dtype=float), 0.0, None)
    total = float(share.sum())
    if total <= 0.0 or not math.isfinite(total):
        return np.full(n, 1.0 / n, dtype=float)
    return np.asarray(share / total, dtype=np.float64)


def refit_stability(
    fit_fn: FitFn,
    data: Sequence[int] | IntArray,
    n_restarts: int,
    seed: int,
    *,
    block_len: int | None = None,
) -> RefitStability:
    """Refit ``fit_fn`` under seeded restarts + block-bootstrap resamples.

    The reference model fits the untouched stream at ``seed``. Each restart
    draws a moving-block bootstrap resample and a fresh child seed, refits,
    and is matched back to the reference by Hungarian assignment on the
    transition-row signatures + emission rows — so restarts vary both the
    data and any seed-driven init, and a mere label permutation costs
    nothing. Reference and refits are first relabeled into a deterministic
    canonical order so assignment indices are comparable across restarts.
    """
    stream = np.asarray(data, dtype=np.intp)
    if stream.ndim != 1 or stream.size == 0:
        raise ValueError("data must be a non-empty 1-D event stream")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    bl = block_len if block_len is not None else max(1, int(round(math.sqrt(stream.size))))
    if bl < 1:
        raise ValueError("block_len must be >= 1")
    ref_raw = fit_fn(stream, int(seed))
    reference = _relabel(ref_raw, _canonical_order(ref_raw))
    rng = np.random.default_rng(int(seed))
    assignments: list[list[int]] = []
    matched: list[DiscreteHMM] = []
    failed = 0
    for _ in range(int(n_restarts)):
        resample = _block_bootstrap(stream, rng, bl)
        child_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        try:
            fitted = fit_fn(resample, child_seed)
        except (ValueError, FloatingPointError, ArithmeticError):
            failed += 1
            continue
        fitted = _relabel(fitted, _canonical_order(fitted))
        perm = _match_states(reference, fitted)
        assignments.append([int(v) for v in perm.tolist()])
        matched.append(_relabel(fitted, perm))
    return RefitStability(
        reference=reference,
        seed=int(seed),
        n_restarts=int(n_restarts),
        block_len=bl,
        stream_sha256=hash_bytes(canonical_json_bytes(stream.tolist())),
        assignments=assignments,
        matched=matched,
        failed=failed,
    )


def stability_report(
    stability: RefitStability,
    *,
    max_dev: float = 0.05,
    min_share: float = _STABLE_SHARE,
) -> dict[str, Any]:
    """Occupancy-share spread and matched A/B Frobenius deviation + verdict.

    Verdict is ``stable`` when the matched-state parameter deviation —
    ``max(||A_m - A_ref||_F, ||B_m - B_ref||_F)`` per restart — stays under
    ``max_dev`` on at least ``min_share`` (default 90%) of restarts; failed
    refits count as unstable restarts.
    """
    ref = stability.reference
    ref_share = _stationary_share(ref.A)
    restarts: list[dict[str, Any]] = []
    shares: list[Array] = []
    for idx, model in enumerate(stability.matched):
        dev_a = float(np.linalg.norm(model.A - ref.A, "fro"))
        dev_b = float(np.linalg.norm(model.B - ref.B, "fro"))
        dev = max(dev_a, dev_b)
        shares.append(_stationary_share(model.A))
        restarts.append(
            {
                "restart": idx,
                "assignment": stability.assignments[idx],
                "dev_a": dev_a,
                "dev_b": dev_b,
                "dev": dev,
                "stable": bool(dev <= max_dev),
            }
        )
    n_stable = sum(1 for row in restarts if row["stable"])
    stable_share = n_stable / stability.n_restarts
    states: list[dict[str, Any]] = []
    for s in range(ref.n_states):
        votes = [a[s] for a in stability.assignments]
        consistency = (max(votes.count(v) for v in set(votes)) / len(votes)) if votes else 0.0
        row_shares = [float(sh[s]) for sh in shares]
        share_spread = max(row_shares) - min(row_shares) if row_shares else 0.0
        a_dev = max(
            (float(np.max(np.abs(m.A[s] - ref.A[s]))) for m in stability.matched), default=0.0
        )
        b_dev = max(
            (float(np.max(np.abs(m.B[s] - ref.B[s]))) for m in stability.matched), default=0.0
        )
        states.append(
            {
                "state": s,
                "ref_share": float(ref_share[s]),
                "assignment_consistency": float(consistency),
                "share_spread": float(share_spread),
                "a_row_max_dev": a_dev,
                "b_row_max_dev": b_dev,
            }
        )
    return {
        "verdict": "stable" if stable_share >= min_share else "unstable",
        "n_restarts": stability.n_restarts,
        "n_matched": len(stability.matched),
        "n_failed": stability.failed,
        "n_stable": n_stable,
        "stable_share": float(stable_share),
        "thresholds": {"max_dev": float(max_dev), "min_share": float(min_share)},
        "max_observed_dev": max((row["dev"] for row in restarts), default=None),
        "block_len": stability.block_len,
        "seed": stability.seed,
        "stream_sha256": stability.stream_sha256,
        "states": states,
        "restarts": restarts,
    }


def _simulate(
    model: DiscreteHMM, n_steps: int, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    """Sample (state path, emissions) from a discrete HMM."""
    states = np.empty(n_steps, dtype=np.intp)
    obs = np.empty(n_steps, dtype=np.intp)
    states[0] = int(rng.choice(model.n_states, p=model.pi))
    obs[0] = int(rng.choice(model.n_obs, p=model.B[states[0]]))
    for t in range(1, n_steps):
        states[t] = int(rng.choice(model.n_states, p=model.A[states[t - 1]]))
        obs[t] = int(rng.choice(model.n_obs, p=model.B[states[t]]))
    return states, obs


def hmm_stability_bench(
    seed: int = 7,
    *,
    n_steps: int = 1500,
    n_restarts: int = 8,
    n_iter: int = 30,
    n_inits: int = 5,
    max_dev: float = 0.25,
    min_share: float = _STABLE_SHARE,
    block_len: int | None = None,
    receipts_dir: Path | str | None = Path("receipts"),
) -> dict[str, Any]:
    """Planted 2-state stream: Baum–Welch must recover the same states across refits.

    Single EM inits are basin-sensitive — roughly half of random inits of the
    planted stream converge to a degenerate decomposition — so the bench's
    fit keeps the highest-likelihood of ``n_inits`` seeded inits, the standard
    best-of-K answer to EM non-convexity. Stability of the *decomposition*
    (the object a regime claim rests on) is then what remains to be measured.

    Returns the sealed ``hmm_stability.v1`` receipt; when ``receipts_dir`` is
    given it is also published immutably to ``<dir>/hmm_stability.json``.
    """
    planted = DiscreteHMM(
        np.array([[0.93, 0.07], [0.09, 0.91]], dtype=float),
        np.array([[0.60, 0.40, 0.00, 0.00], [0.00, 0.00, 0.40, 0.60]], dtype=float),
        np.array([0.5, 0.5], dtype=float),
    )
    rng = np.random.default_rng(int(seed))
    truth_states, stream = _simulate(planted, int(n_steps), rng)

    def fit_fn(obs: IntArray, fit_seed: int) -> DiscreteHMM:
        best: DiscreteHMM | None = None
        best_ll = -math.inf
        for child in np.random.SeedSequence(fit_seed).spawn(max(1, int(n_inits))):
            model, history = baum_welch(
                obs,
                n_states=2,
                n_obs=4,
                n_iter=int(n_iter),
                seed=int(child.generate_state(1)[0]),
                log_space=True,
            )
            if history[-1] > best_ll:
                best, best_ll = model, history[-1]
        if best is None:  # pragma: no cover - spawn(max(1, .)) is never empty
            raise ValueError("n_inits must be >= 1")
        return best

    stability = refit_stability(fit_fn, stream, int(n_restarts), int(seed), block_len=block_len)
    report = stability_report(stability, max_dev=max_dev, min_share=min_share)
    # Recovery of the planted decomposition: match the reference fit to the
    # truth, then score its Viterbi path against the planted state path.
    truth_perm = _match_states(planted, stability.reference)
    ref_in_truth = _relabel(stability.reference, truth_perm)
    path, _ = log_viterbi(ref_in_truth, stream)
    recovery = {
        "state_path_agreement": float(np.mean(path == truth_states)),
        "truth_dev_a": float(np.linalg.norm(ref_in_truth.A - planted.A, "fro")),
        "truth_dev_b": float(np.linalg.norm(ref_in_truth.B - planted.B, "fro")),
        "truth_assignment": [int(v) for v in truth_perm.tolist()],
    }
    receipt: dict[str, Any] = {
        "kind": "hmm_stability",
        "schema": HMM_STABILITY_SCHEMA,
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "refit_stability_diagnostic",
        "interpretation": str(report["verdict"]),
        "seed": int(seed),
        "inputs_sha256": stability.stream_sha256,
        "params": {
            "n_steps": int(n_steps),
            "n_states": 2,
            "n_obs": 4,
            "n_restarts": int(n_restarts),
            "n_iter": int(n_iter),
            "n_inits": int(n_inits),
            "block_len": stability.block_len,
            "max_dev": float(max_dev),
            "min_share": float(min_share),
        },
        "planted": planted.as_dict(),
        "recovery": recovery,
        "report": report,
        "verdict": str(report["verdict"]),
        "evidence": [
            "planted_two_state_stream",
            "block_bootstrap_restarts",
            "hungarian_state_matching",
            "label_switching_invariant",
        ],
    }
    receipt["receipt_sha256"] = hash_bytes(canonical_json_bytes(receipt))
    payload: dict[str, Any] = json.loads(canonical_json_bytes(receipt))
    if receipts_dir is not None:
        publish_text_once(
            Path(receipts_dir) / "hmm_stability.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    return payload


__all__ = [
    "HMM_STABILITY_SCHEMA",
    "FitFn",
    "RefitStability",
    "hmm_stability_bench",
    "refit_stability",
    "stability_report",
]
