"""Member attribution for the OOF quantile stack.

A pinned stack weight is evidence a member *can* contribute; it is not
evidence it *does*. This module measures realized contribution under the
``cross_fitted_quantile_stack`` oracle:

* ``leave_one_out_deltas`` — drop each member, restack, and read the pinball
  loss increase. A member that earns its weight leaves a positive delta.
* ``shapley_members`` — Shapley values over member subsets with the
  negative pinball loss as the coalition value; exact for small
  committees, seeded permutation sampling otherwise.
* ``phantom_members`` — flags members whose attribution is
  indistinguishable from a shuffled-copy null: weight > 0 but
  contribution inside the noise floor.

The shuffled-member null is the honest baseline — a phantom head can
absorb real weight by looking mildly correlated with the target.
"""

from __future__ import annotations

import itertools
from math import factorial
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.fusion.quantile_stack import cross_fitted_quantile_stack
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_EXACT_SHAPLEY_MAX = 10
_N_PERMS_DEFAULT = 64
_PHANTOM_QUANTILE = 0.95


def _oof_pinball(
    q: NDArray[np.float64],
    y: NDArray[np.float64],
    taus: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    alpha: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """(mean pinball, per-tau pinball, OOF weights) for a member subset."""
    res = cross_fitted_quantile_stack(q, y, taus, folds, alpha=alpha)
    eligible = res.fold_ids >= 0
    resid = y[eligible, None] - res.oof_quantiles[eligible]
    per_tau = np.array(
        [float(np.mean(resid[:, j] * (taus[j] - (resid[:, j] < 0.0)))) for j in range(taus.size)]
    )
    return float(per_tau.mean()), per_tau, res.weights


def leave_one_out_deltas(
    member_quantiles: NDArray[np.float64],
    target: NDArray[np.float64],
    taus: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Per-member stack-loss delta from dropping that member.

    ``delta[k] = loo_loss[k] - full_loss``: positive means the member
    earned its weight on out-of-fold rows.
    """
    q = np.asarray(member_quantiles, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    t = np.asarray(taus, dtype=float).reshape(-1)
    n, m, k = q.shape
    if m < 2:
        raise ValueError("attribution needs at least two members")
    full, full_tau, weights = _oof_pinball(q, y, t, folds, alpha)
    deltas = np.zeros(m)
    deltas_tau = np.zeros((m, k))
    for i in range(m):
        keep = [j for j in range(m) if j != i]
        loo, loo_tau, _ = _oof_pinball(q[:, keep, :], y, t, folds, alpha)
        deltas[i] = loo - full
        deltas_tau[i] = loo_tau - full_tau
    return {
        "full_pinball": full,
        "deltas": deltas,
        "deltas_per_tau": deltas_tau,
        "weights": weights,
    }


def _subset_value(
    q: NDArray[np.float64],
    y: NDArray[np.float64],
    t: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    alpha: float,
    subset: tuple[int, ...],
    cache: dict[tuple[int, ...], float],
) -> float:
    key = tuple(sorted(subset))
    if key in cache:
        return cache[key]
    if not key:
        # empty coalition: unconditional-quantile loss per tau
        per_tau = np.array(
            [
                float(
                    np.mean((y - np.quantile(y, tau)) * (tau - ((y - np.quantile(y, tau)) < 0.0)))
                )
                for tau in t
            ]
        ).mean()
        value = -float(per_tau)
    else:
        cols = np.array(key, dtype=int)
        loss, _, _ = _oof_pinball(q[:, cols, :], y, t, folds, alpha)
        value = -loss
    cache[key] = value
    return value


def shapley_members(
    member_quantiles: NDArray[np.float64],
    target: NDArray[np.float64],
    taus: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    *,
    alpha: float = 1.0,
    n_perms: int = _N_PERMS_DEFAULT,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Shapley attribution with coalition value = -OOF pinball loss.

    Exact enumeration for ``m <= 10``; seeded permutation sampling
    (deterministic) above that.
    """
    q = np.asarray(member_quantiles, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    t = np.asarray(taus, dtype=float).reshape(-1)
    m = q.shape[1]
    if m < 2:
        raise ValueError("attribution needs at least two members")
    cache: dict[tuple[int, ...], float] = {}
    phi = np.zeros(m)
    if m <= _EXACT_SHAPLEY_MAX:
        weight_norm = factorial(m)
        for i in range(m):
            others = [j for j in range(m) if j != i]
            for s in range(len(others) + 1):
                w = factorial(s) * factorial(m - s - 1) / weight_norm
                for subset in itertools.combinations(others, s):
                    with_i = _subset_value(q, y, t, folds, alpha, (*subset, i), cache)
                    without = _subset_value(q, y, t, folds, alpha, subset, cache)
                    phi[i] += w * (with_i - without)
        return phi

    rng = np.random.default_rng(seed)
    for _ in range(max(1, n_perms)):
        perm = rng.permutation(m)
        prefix: tuple[int, ...] = ()
        prev = _subset_value(q, y, t, folds, alpha, prefix, cache)
        for i in perm:
            prefix = tuple(sorted((*prefix, int(i))))
            cur = _subset_value(q, y, t, folds, alpha, prefix, cache)
            phi[int(i)] += cur - prev
            prev = cur
    return phi / max(1, n_perms)


def phantom_members(
    member_quantiles: NDArray[np.float64],
    target: NDArray[np.float64],
    taus: NDArray[np.float64],
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    *,
    alpha: float = 1.0,
    n_null: int = 8,
    seed: int = 0,
    quantile: float = _PHANTOM_QUANTILE,
) -> dict[str, Any]:
    """Flag members whose LOO contribution is inside the noise floor.

    Null: shuffle member ``k``'s rows independently (destroying its target
    alignment while preserving its marginal distribution), re-measure its
    LOO delta. The floor is the ``quantile`` of the pooled |null deltas|;
    a weighted member below it is phantom.
    """
    base = leave_one_out_deltas(member_quantiles, target, taus, folds, alpha=alpha)
    q = np.asarray(member_quantiles, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    t = np.asarray(taus, dtype=float).reshape(-1)
    m = q.shape[1]
    rng = np.random.default_rng(seed)
    null_deltas = np.zeros((n_null, m))
    for rep in range(n_null):
        qs = q.copy()
        for i in range(m):
            qs[:, i, :] = qs[rng.permutation(q.shape[0]), i, :]
        null_deltas[rep] = leave_one_out_deltas(qs, y, t, folds, alpha=alpha)["deltas"]
    floor = float(np.quantile(np.abs(null_deltas), quantile))
    w = base["weights"]
    weight_mask = np.asarray(w.mean(axis=0) > 0.0) if w.ndim == 2 else np.zeros(m, dtype=bool)
    phantom = np.array([(base["deltas"][i] <= floor) and weight_mask[i] for i in range(m)])
    return {
        "full_pinball": base["full_pinball"],
        "deltas": base["deltas"],
        "null_floor": floor,
        "weighted": weight_mask,
        "phantom": phantom,
    }


def _synthetic_committee(
    n: int, m_signal: int, m_noise: int, n_taus: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``y = x + sigma*eps``; signal heads emit the true conditional
    quantiles of ``y | x`` plus head noise; noise heads emit marginals."""
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.1, 0.9, n_taus)
    sigma = 0.5
    x = rng.normal(0.0, 1.0, n)
    y = x + sigma * rng.normal(0.0, 1.0, n)
    q_true = x[:, None] + sigma * norm.ppf(taus)[None, :]
    m = m_signal + m_noise
    q = np.zeros((n, m, n_taus))
    for i in range(m_signal):
        q[:, i, :] = q_true + rng.normal(0.0, 0.05 + 0.10 * i, (n, n_taus))
    for i in range(m_noise):
        q[:, m_signal + i, :] = rng.normal(0.0, 1.0, (n, n_taus))
    return q, y, taus


def _chronological_folds(n: int, n_folds: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Contiguous chronological folds: train on all rows before the test block."""
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds = []
    for f in range(1, n_folds):
        train = np.arange(0, edges[f])
        test = np.arange(edges[f], edges[f + 1])
        if train.size >= 2 and test.size >= 1:
            folds.append((train.astype(np.int64), test.astype(np.int64)))
    return folds


def attribution_bench(
    n: int = 200,
    m_signal: int = 4,
    m_noise: int = 2,
    n_taus: int = 9,
    n_folds: int = 5,
    seed: int = 0,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Plant a noise committee and check phantom detection + attribution sign."""
    q, y, taus = _synthetic_committee(n, m_signal, m_noise, n_taus, seed)
    folds = _chronological_folds(n, n_folds)
    loo = leave_one_out_deltas(q, y, taus, folds, alpha=alpha)
    ph = phantom_members(q, y, taus, folds, alpha=alpha, seed=seed)
    phi = shapley_members(q, y, taus, folds, alpha=alpha, seed=seed)
    m = m_signal + m_noise
    true_phantom = np.zeros(m, dtype=bool)
    true_phantom[m_signal:] = True
    detected = ph["phantom"]
    tp = int(np.count_nonzero(detected & true_phantom))
    fp = int(np.count_nonzero(detected & ~true_phantom))
    fn = int(np.count_nonzero(~detected & true_phantom))
    interpretation = {
        "n": n,
        "m_signal": m_signal,
        "m_noise": m_noise,
        "full_pinball": loo["full_pinball"],
        "loo_deltas": [float(x) for x in loo["deltas"]],
        "signal_delta_mean": float(loo["deltas"][:m_signal].mean()),
        "noise_delta_mean": float(loo["deltas"][m_signal:].mean()),
        "shapley": [float(x) for x in phi],
        "noise_floor": ph["null_floor"],
        "phantom_detected": [int(i) for i in np.nonzero(detected)[0]],
        "phantom_truth": [int(i) for i in np.nonzero(true_phantom)[0]],
        "confusion": {"tp": tp, "fp": fp, "fn": fn},
    }
    payload: dict[str, Any] = {
        "kind": "attribution",
        "schema": "attribution.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "phantom members get flagged; signal members earn positive LOO delta",
            "verdict": "ok" if fn == 0 and fp == 0 else "partial",
        },
        "interpretation": interpretation,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
