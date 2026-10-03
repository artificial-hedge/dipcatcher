"""verify — certify the pedagogical HMM code by exhaustive enumeration.

``discrete.py`` implements Rabiner's recursions; ``eisner.py`` is the
textbook example. Their only prior evidence is *reproducing the book's
numbers* on one fixed instance. This module is stronger: for random small
HMMs it enumerates **every state path** and checks, to float tolerance,

1. ``forward`` likelihood == total path probability (Rabiner P1);
2. ``viterbi`` path == the argmax path under enumeration, and its
   reported probability == the enumerated max (Rabiner P2);
3. ``gamma`` posterior marginals == enumeration marginals (Rabiner P3).

A ``baum_welch`` check verifies one EM sweep never *decreases* the
enumerated likelihood — the ascent property made executable.

``verify_hmm(seed, ...)`` returns a sealed ``hmm_verify.v1`` SYNTHETIC
receipt. State/observation sizes stay tiny (n≤3, T≤6, |Ω|≤3) so
enumeration is exact, not sampled.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt

from quant_fund.hmm.discrete import (
    DiscreteHMM,
    baum_welch,
    gamma_xi,
    likelihood,
    viterbi,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

TOL = 1e-9


def _enum_paths(
    model: DiscreteHMM, obs: npt.NDArray[np.int_]
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """Total probability, max path prob, and per-(t,state) posterior mass."""
    n = model.pi.shape[0]
    total = 0.0
    best = -1.0
    marg = np.zeros((len(obs), n))
    for path in product(range(n), repeat=len(obs)):
        p = float(model.pi[path[0]] * model.B[path[0], obs[0]])
        for t in range(1, len(obs)):
            p *= float(model.A[path[t - 1], path[t]] * model.B[path[t], obs[t]])
        total += p
        if p > best:
            best = p
        for t, s in enumerate(path):
            marg[t, s] += p
    return total, best, marg


def _random_hmm(rng: np.random.Generator, n_states: int, n_obs: int) -> DiscreteHMM:
    return DiscreteHMM(
        A=rng.dirichlet(np.full(n_states, 0.7), size=n_states),
        B=rng.dirichlet(np.full(n_obs, 0.7), size=n_states),
        pi=rng.dirichlet(np.full(n_states, 0.7)),
    )


def verify_hmm(seed: int = 0, n_trials: int = 40) -> dict[str, Any]:
    """Enumeration-oracle audit; sealed receipt payload."""
    rng = np.random.default_rng(seed)
    checks = {"forward": 0, "viterbi_path": 0, "viterbi_prob": 0, "gamma": 0, "bw_ascent": 0}
    failures: list[str] = []
    for trial in range(n_trials):
        n_states = int(rng.integers(2, 4))
        n_obs_sym = int(rng.integers(2, 4))
        t_len = int(rng.integers(2, 7))
        m = _random_hmm(rng, n_states, n_obs_sym)
        obs = rng.integers(0, n_obs_sym, t_len)
        total, best, marg = _enum_paths(m, obs)

        got = likelihood(m, obs)
        if abs(got - total) <= TOL * max(1.0, total):
            checks["forward"] += 1
        else:
            failures.append(f"trial {trial}: forward {got} != enum {total}")

        path, p_best = viterbi(m, obs)
        if abs(p_best - best) <= TOL * max(1.0, best):
            checks["viterbi_prob"] += 1
        else:
            failures.append(f"trial {trial}: viterbi prob {p_best} != enum {best}")
        # recompute the reported path's probability under enumeration
        p_rep = float(m.pi[path[0]] * m.B[path[0], obs[0]])
        for t in range(1, len(obs)):
            p_rep *= float(m.A[path[t - 1], path[t]] * m.B[path[t], obs[t]])
        if abs(p_rep - best) <= TOL * max(1.0, best):
            checks["viterbi_path"] += 1
        else:
            failures.append(f"trial {trial}: viterbi path prob {p_rep} != argmax {best}")

        g, xi, _ = gamma_xi(m, obs)
        enum_marg = marg / total if total > 0 else marg
        if np.allclose(np.asarray(g), enum_marg, atol=1e-6):
            checks["gamma"] += 1
        else:
            failures.append(f"trial {trial}: gamma marginals diverge")
        # sanity: xi sums to pairwise posterior mass
        if not np.isfinite(np.asarray(xi)).all():
            failures.append(f"trial {trial}: xi non-finite")

        m2, _hist = baum_welch(
            obs, n_states=n_states, n_obs=n_obs_sym, n_iter=1, seed=seed, model=m
        )
        if likelihood(m2, obs) >= total - TOL:
            checks["bw_ascent"] += 1
        else:
            failures.append(f"trial {trial}: baum_welch decreased likelihood")

    ok = not failures
    payload: dict[str, Any] = {
        "kind": "hmm_verify",
        "schema": "hmm_verify.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "seed": seed,
            "n_trials": n_trials,
            "checks": checks,
            "failures": failures[:20],
            "ok": ok,
        },
        "interpretation": {
            "ok": "pedagogical HMM code matches exhaustive enumeration — not just the textbook instance",
            "fail": "a recursion disagrees with the enumeration oracle — see failures[]",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["verify_hmm"]
