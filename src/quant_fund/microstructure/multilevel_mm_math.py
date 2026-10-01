"""Pure allocation and reward math for synthetic multilevel market making.

The numerical transforms, Hamilton apportionment, and potential shaping are
independent of simulator state and optional neural-network dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


def _finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _prob(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {x!r}")
    return v


def _int_at_least(x: int, floor: int, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < floor:
        raise ValueError(f"{name} must be an int >= {floor}, got {x!r}")
    return x


def _seed_int(x: int) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise ValueError(f"seed must be a non-negative int, got {x!r}")
    return int(x)


def hamilton_apportionment(weights: Sequence[float] | Array, lots: int) -> IntArray:
    """Round simplex weights to an integer lot split (paper Sec. 3.2).

    Hamilton / largest-remainder method (Balinski & Young 2010): quotas
    ``w_k * M`` are floored and the remaining lots are dealt one each to the
    largest fractional remainders, ties broken by the lowest component index
    (deterministic). Weights must be non-negative and finite with positive
    sum; they are renormalized so the output always sums to ``lots``.
    """
    m = _int_at_least(lots, 1, "lots")
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be non-negative, got {weights!r}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    w = w / total
    quotas = w * m
    base = np.floor(quotas).astype(np.int64)
    remainder = m - int(base.sum())
    order = np.argsort(-(quotas - base), kind="stable")
    for i in range(remainder):
        base[int(order[i])] += 1
    return base


def simplex_transform(x: Sequence[float] | Array) -> Array:
    """Logistic map ``h: R^d -> S^{d+1}`` (paper Eq. before 8).

    ``a_k = exp(x_k) / (1 + sum_l exp(x_l))`` for ``k = 1..d`` and the
    reference component ``a_0 = 1 / (1 + sum_l exp(x_l))``. The output is a
    strictly-positive point of the open simplex (all components in (0, 1),
    summing to 1). Computed in a max-shifted form for numerical stability.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("x must be finite")
    z = np.concatenate(([0.0], v))
    z = z - float(z.max())
    e = np.exp(z)
    return e / float(e.sum())


def alr_transform(a: Sequence[float] | Array) -> Array:
    """Additive log-ratio inverse of :func:`simplex_transform`.

    ``y_k = log(a_k / a_0)`` for ``k = 1..d`` — the exact inverse of ``h`` on
    the open simplex, so ``alr_transform(simplex_transform(x)) == x``
    componentwise. Fail closed: entries must be strictly positive (the open
    simplex) and sum to 1.
    """
    v = np.asarray(a, dtype=np.float64).ravel()
    if v.size < 2:
        raise ValueError(f"a must have >= 2 components, got {v.size}")
    if not np.all(np.isfinite(v)):
        raise ValueError("a must be finite")
    if np.any(v <= 0.0):
        raise ValueError("a must be strictly positive (open simplex)")
    s = float(v.sum())
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"a must sum to 1, got sum={s!r}")
    return np.asarray(np.log(v[1:] / v[0]), dtype=np.float64)


def logistic_normal_logpdf(a: Sequence[float] | Array, mean: Array, logvar: Array) -> float:
    """Closed-form logistic-normal log density (Aitchison & Shen 1980).

    If ``X ~ N(mu, Diag(exp(logvar)))`` then ``a = h(X)`` has density

        p(a) = N(alr(a); mu, Sigma) / prod_{k=0}^{d} a_k ,

    the Jacobian of the ALR map being ``prod_k a_k`` over all ``d + 1``
    components including the reference ``a_0``. This is the density behind the
    policy-gradient term ``log pi_theta(a|s)`` of the paper's Eq. 12/14.
    """
    y = alr_transform(a)
    mu = np.asarray(mean, dtype=np.float64).ravel()
    lv = np.asarray(logvar, dtype=np.float64).ravel()
    d = y.size
    if mu.shape != (d,) or lv.shape != (d,):
        raise ValueError(f"mean/logvar must have shape ({d},), got {mu.shape} and {lv.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(lv)):
        raise ValueError("mean and logvar must be finite")
    av = np.asarray(a, dtype=np.float64).ravel()
    var = np.exp(lv)
    resid = (y - mu) ** 2 / var
    log_n = -0.5 * (d * math.log(2.0 * math.pi) + float(lv.sum()) + float(resid.sum()))
    return float(log_n - float(np.log(av).sum()))


def logistic_normal_sample(mean: Array, logvar: Array, rng: np.random.Generator) -> Array:
    """Draw ``a = h(X)`` with ``X ~ N(mu, Diag(exp(logvar)))``; seeded RNG."""
    mu = np.asarray(mean, dtype=np.float64).ravel()
    lv = np.asarray(logvar, dtype=np.float64).ravel()
    if mu.size == 0:
        raise ValueError("mean must be non-empty")
    if mu.shape != lv.shape:
        raise ValueError(f"mean and logvar shapes must agree, got {mu.shape} / {lv.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(lv)):
        raise ValueError("mean and logvar must be finite")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    z = rng.standard_normal(mu.size)
    return simplex_transform(mu + np.exp(0.5 * lv) * z)


def potential_shaping(phi_now: float, phi_next: float, gamma: float = 1.0) -> float:
    """Shaping bonus ``F = gamma * Phi(s') - Phi(s)``.

    The paper's per-decision reward carries ``Phi(s) = Q p`` (inventory valued
    at mid) at ``gamma = 1`` — the ``Q_{n+1} p_{n+1} - Q_n p_n`` term of Eq. 5.
    Fail closed on non-finite potentials or a discount outside ``(0, 1]``.
    """
    p0 = _finite(phi_now, "phi_now")
    p1 = _finite(phi_next, "phi_next")
    g = float(gamma)
    if not math.isfinite(g) or not 0.0 < g <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma!r}")
    return g * p1 - p0


def shape_episode_rewards(
    rewards: Sequence[float] | Array,
    potentials: Sequence[float] | Array,
    gamma: float = 1.0,
) -> Array:
    """Apply potential shaping to a length-``N`` reward sequence.

    ``potentials`` has length ``N + 1`` (``Phi(s_0) .. Phi(s_N)``); returns
    ``r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t)``. Invariance (documented,
    tested): the discounted shaped return is ``R' = R + gamma^N Phi(s_N) -
    Phi(s_0)`` — an offset independent of the action sequence, so the optimal
    policy class is preserved.
    """
    r = np.asarray(rewards, dtype=np.float64).ravel()
    phi = np.asarray(potentials, dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("rewards must be non-empty")
    if phi.size != r.size + 1:
        raise ValueError(
            f"potentials must have length n_rewards + 1 ({r.size + 1}), got {phi.size}"
        )
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(phi)):
        raise ValueError("rewards and potentials must be finite")
    g = float(gamma)
    if not math.isfinite(g) or not 0.0 < g <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma!r}")
    return r + g * phi[1:] - phi[:-1]


def shaped_optimal_q(q_values: Sequence[float] | Array, phi_state: float) -> Array:
    """Optimal Q under the shaped reward: ``Q'*(s, a) = Q*(s, a) - Phi(s)``.

    Ng-Harada-Russell (1999) invariance: adding ``gamma*Phi(s') - Phi(s)`` to
    every reward shifts the optimal action-value at ``s`` by the constant
    ``-Phi(s)``, so ``argmax_a`` is exactly preserved. This helper implements
    the shift; tests assert the argmax invariance.
    """
    q = np.asarray(q_values, dtype=np.float64).ravel()
    if q.size == 0:
        raise ValueError("q_values must be non-empty")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_values must be finite")
    p = _finite(phi_state, "phi_state")
    return q - p


def mm_reward(
    *,
    cash_flow: float,
    q_prev: int,
    p_prev: float,
    q_next: int,
    p_next: float,
    inv_gamma: float,
    lots: int,
) -> float:
    """Per-decision reward, paper Eq. 5.

    ``r = (cash_flow + (Q_{n+1} p_{n+1} - Q_n p_n) - gamma |Q_{n+1}|) / M``
    where ``cash_flow`` is signed fill cash over ``(t_n, t_{n+1}]``, ``Q`` the
    lot inventory, ``p`` the mid. The ``Q'p' - Qp`` term is the potential
    shaping ``Phi = Q p`` (telescopes over the episode, Eq. 7); the
    ``-gamma |Q'|`` term is the inventory penalty. Simulator-internal
    (``sim_internal_*``) training signal, never a headline metric.
    """
    cf = _finite(cash_flow, "cash_flow")
    p0 = _pos_finite(p_prev, "p_prev")
    p1 = _pos_finite(p_next, "p_next")
    g = _nonneg_finite(inv_gamma, "inv_gamma")
    m = _int_at_least(lots, 1, "lots")
    if isinstance(q_prev, bool) or isinstance(q_next, bool):
        raise ValueError("inventories must be ints")
    return float((cf + (q_next * p1 - q_prev * p0) - g * abs(q_next)) / m)


def terminal_position_limit(lots: int, nu: float) -> int:
    """Paper Sec. 3.3 terminal bound ``ceil(nu * M)``, ``nu in [0, 1]``."""
    m = _int_at_least(lots, 1, "lots")
    v = _prob(nu, "nu")
    return int(math.ceil(v * m))


def terminal_reward(
    *,
    q_n: int,
    p_n: float,
    q_plus: int,
    mo_cash: float,
    lots: int,
) -> float:
    """Terminal reward ``g(s_N)``, paper Eq. 6.

    Inventory beyond ``ceil(nu M)`` is liquidated by a market order; ``mo_cash``
    is the signed cash actually received (it walks the book), and the leftover
    ``Q_{N+}`` is valued at the terminal mid: ``g = (p_N (Q_{N+} - Q_N) +
    MO_nu) / M``. Fail closed on inconsistent liquidation accounting
    (``|q_plus|`` must not exceed ``|q_n|`` with matching sign).
    """
    p = _pos_finite(p_n, "p_n")
    c = _finite(mo_cash, "mo_cash")
    m = _int_at_least(lots, 1, "lots")
    if isinstance(q_n, bool) or isinstance(q_plus, bool):
        raise ValueError("inventories must be ints")
    if abs(q_plus) > abs(q_n) or (q_plus != 0 and (q_plus > 0) != (q_n > 0)):
        raise ValueError(
            f"q_plus must be a partial liquidation of q_n, got q_n={q_n}, q_plus={q_plus}"
        )
    return float((p * (q_plus - q_n) + c) / m)


def deep_set_pool(features: Sequence[Sequence[float]] | Array, slots: Array, n_slots: int) -> Array:
    """Mean-pool element embeddings into ``n_slots`` level buckets (flattened).

    Per Zaheer et al. (2017) and the paper's Sec. 4.1: each element is embedded
    by a shared ``f^o_phi`` and the embeddings inside each price level are
    averaged; empty levels map to zeros. Permutation-invariant by construction:
    rows inside each slot are canonically sorted before averaging, so
    shuffling the rows of ``(features, slots)`` leaves the output bitwise
    identical — this is the property the tests assert.
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f.reshape(-1, 1) if f.size else np.zeros((0, 1))
    if f.ndim != 2:
        raise ValueError(f"features must be 2-D (n_elems, d), got shape {f.shape}")
    s = np.asarray(slots, dtype=np.int64).ravel()
    if s.shape[0] != f.shape[0]:
        raise ValueError(f"slots length {s.shape[0]} must match n_elems {f.shape[0]}")
    k = _int_at_least(n_slots, 1, "n_slots")
    if s.size and (int(s.min()) < 0 or int(s.max()) >= k):
        raise ValueError(f"slots must lie in [0, {k}), got range {s.min()}..{s.max()}")
    if not np.all(np.isfinite(f)):
        raise ValueError("features must be finite")
    pooled = np.zeros((k, f.shape[1]), dtype=np.float64)
    if s.size:
        for j in range(k):
            rows = f[s == j]
            if rows.shape[0]:
                # Canonical (lexicographic) row order inside a slot: the mean
                # is mathematically unchanged but bitwise identical under
                # element permutations — determinism the tests pin down.
                order = np.lexsort(rows.T[::-1])
                pooled[j] = rows[order].mean(axis=0)
    return pooled.reshape(-1)
