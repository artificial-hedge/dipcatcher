"""White Reality Check and Hansen SPA for data-snooping.

When K strategies are screened on the same sample, the best
observed mean excess score is biased upward. The Reality Check
bootstraps the max-statistic; Hansen's SPA refines it with
a studentized, loss-weighted bootstrap that is less sensitive
to irrelevant (bad) alternatives.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure size/power on generated
score panels — never evidence about real strategies, and the
bench uses proper scores only.

References:
- White, H. (2000). A reality check for data snooping.
  *Econometrica* 68, 1097-1126 — bootstrap of max over
  centered statistics.
- Hansen, P. R. (2005). A test for superior predictive
  ability. *Journal of Business & Economic Statistics* 23 —
  SPA: studentized statistic, thresholded recentering.
- Romano, J. P., Wolf, M. (2005). Stepwise multiple testing
  as formalized data snooping. *Econometrica* 73 — stepdown
  refinement quoted alongside.
- Sullivan, R., Timmermann, A., White, H. (1999). Data-snooping,
  technical trading rule performance. *Journal of Finance* 54.

Composition: pure numpy + scipy — stationary block bootstrap
on centered score differences; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _stationary_boot(t: int, b: float, rng: np.random.Generator) -> NDArray[np.int64]:
    """Stationary bootstrap indices (Politis-Romano, geom block
    lengths with mean b)."""
    idx = np.empty(t, dtype=np.int64)
    idx[0] = rng.integers(0, t)
    i = 1
    while i < t:
        if rng.uniform() < 1.0 / b:
            idx[i] = rng.integers(0, t)
        else:
            idx[i] = (idx[i - 1] + 1) % t
        i += 1
    return idx


def white_reality(
    scores: FloatArray,
    block: float = 10.0,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Reality Check: p-value that the best strategy beats 0.

    scores is (T, K) of per-period score *advantages* over the
    benchmark (e.g. loss differences); positive mean = beats."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 50 or s.shape[1] < 2:
        raise ValueError("scores (T>=50, K>=2) required")
    if not np.all(np.isfinite(s)):
        raise ValueError("finite scores required")
    t, k = s.shape
    rng = np.random.default_rng(seed)

    means = s.mean(axis=0)
    best = float(np.max(means) * np.sqrt(t))
    # bootstrap of max of centered means
    null_stats = np.empty(n_boot)
    for r in range(n_boot):
        idx = _stationary_boot(t, block, rng)
        sb = s[idx]
        null_stats[r] = np.max((sb.mean(axis=0) - means) * np.sqrt(t))
    p_rc = float(np.mean(null_stats >= best))
    return {
        "t": float(t),
        "k": float(k),
        "best_mean": float(np.max(means)),
        "rc_stat": best,
        "p_rc": p_rc,
        "n_active": float(np.sum(means > 0)),
    }


def hansen_spa(
    scores: FloatArray,
    block: float = 10.0,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """SPA test: studentized max with thresholded recentering."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 50 or s.shape[1] < 2:
        raise ValueError("scores (T>=50, K>=2) required")
    if not np.all(np.isfinite(s)):
        raise ValueError("finite scores required")
    t, k = s.shape
    rng = np.random.default_rng(seed)

    means = s.mean(axis=0)
    sds = s.std(axis=0, ddof=1)
    # studentized
    tstats = means * np.sqrt(t) / np.clip(sds, 1e-12, None)
    spa_stat = float(np.max(np.maximum(tstats, 0.0)))
    # threshold: only models whose mean is plausibly ~0 get recentered
    thresh = -np.sqrt(2 * np.log(np.log(t))) * sds / np.sqrt(t)
    mu_b = np.where(means <= thresh, means, np.maximum(means, 0.0))
    mu_b = np.minimum(mu_b, 0.0)  # Hansen: center at min(mean, 0)

    null_stats = np.empty(n_boot)
    for r in range(n_boot):
        idx = _stationary_boot(t, block, rng)
        sb = s[idx]
        zb = (sb.mean(axis=0) - mu_b - (means - mu_b)) * np.sqrt(t)
        zb = zb / np.clip(sds, 1e-12, None)
        null_stats[r] = np.max(np.maximum(zb, 0.0))
    p_spa = float(np.mean(null_stats >= spa_stat))
    return {
        "t": float(t),
        "k": float(k),
        "best_t": float(np.max(tstats)),
        "spa_stat": spa_stat,
        "p_spa": p_spa,
        "n_recen": float(np.sum(means <= thresh)),
    }


def synth_snooping(
    t: int = 300,
    k: int = 40,
    true_edges: int = 0,
    edge: float = 0.15,
    ar: float = 0.3,
    seed: int = 0,
) -> FloatArray:
    """K strategy score panels; true_edges carry positive mean
    advantage, rest are pure AR(1) noise."""
    rng = np.random.default_rng(seed)
    s = np.zeros((t, k))
    for j in range(k):
        e = rng.normal(0, 1, t)
        for i in range(1, t):
            e[i] += ar * e[i - 1]
        m = edge if j < true_edges else 0.0
        s[:, j] = m + e / np.sqrt(1 - ar**2)
    return s


def bench_white_reality(seed: int = 20261231 + 250) -> dict[str, float]:
    """Reality-Check/SPA self-check: with 3 true edges the tests
    reject (p small); under pure noise they do not reject hard.
    All ``synthetic_*``."""
    s1 = synth_snooping(true_edges=3, edge=0.2, seed=seed)
    rc = white_reality(s1, n_boot=300, seed=seed)
    spa = hansen_spa(s1, n_boot=300, seed=seed)
    s0 = synth_snooping(true_edges=0, seed=seed + 1)
    rc0 = white_reality(s0, n_boot=300, seed=seed + 2)
    spa0 = hansen_spa(s0, n_boot=300, seed=seed + 3)
    rc_b = white_reality(s1, n_boot=300, seed=seed)

    p1 = float(rc["p_rc"])
    return {
        "synthetic_p_rc": p1,
        "synthetic_p_spa": float(spa["p_spa"]),
        "synthetic_p_rc_null": float(rc0["p_rc"]),
        "synthetic_p_spa_null": float(spa0["p_spa"]),
        "synthetic_best_t": float(spa["best_t"]),
        "synthetic_n_active": float(rc["n_active"]),
        "synthetic_detects": float(
            p1 < 0.1 and float(spa["p_spa"]) < 0.1 and float(rc0["p_rc"]) > p1
        ),
        "synthetic_determinism": float(p1 == float(rc_b["p_rc"])),
    }
