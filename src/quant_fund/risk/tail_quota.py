"""Tail-mass quota watch — anytime-valid alarm that the return tail
is fatter than declared.

``decay_watch`` monitors drift (E[r] >= 0); this lane monitors the tail
quantile itself. Null: ``P(r_t <= c) <= alpha`` at level ``c`` — the
declared return floor breaches at most ``alpha`` of the time.

The e-variable is a likelihood ratio between a predictable alternative
``p_t`` (the Beta-posterior mean breach rate fitted on ``z_1..z_{t-1}``,
Laplace-smoothed) and the null rate ``alpha``:

    e_t = alt_lik(z_t) / null_lik(z_t)
       = (z_t p_t + (1 - z_t)(1 - p_t)) / (z_t alpha + (1 - z_t)(1 - alpha))

A point-null likelihood ratio has E_null[e_t] = 1 exactly, and the
product stays an e-process under any predictable alternative sequence —
so ``prod e_t >= 1/alpha_alarm`` is an anytime-valid rejection of the
tail quota at level ``alarm_at``.

This is a risk-monitoring device, not a performance headline: it emits
the e-value path and alarm events, never P&L or Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def tail_quota_eprocess(
    returns: Array,
    *,
    c: float,
    alpha: float,
    init_prior: float | None = None,
    alarm_at: float = 100.0,
) -> dict[str, Any]:
    """Likelihood-ratio e-process for H0: breach rate <= alpha at level c.

    ``c`` is the declared return floor (e.g. -0.02); ``alpha`` its
    declared breach rate. The alternative estimate ``p_t`` is the
    online breach-rate posterior mean under a Beta(1,1) prior — fully
    predictable, so the product process is an e-process under H0.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be a 1-d array of length >= 2")
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite")
    if not (0 < alpha < 1) or not np.isfinite(c) or alarm_at <= 0:
        raise ValueError("need 0<alpha<1, finite c, alarm_at>0")
    z = (r <= c).astype(np.float64)
    e = 1.0
    path = [1.0]
    alarm_t: int | None = None
    n_breach = 0
    # p_t fitted on z_1..z_{t-1} — Beta posterior mean, Laplace-smoothed
    for t, zt in enumerate(z):
        if t == 0 and init_prior is not None:
            p0 = init_prior
        else:
            p0 = (1.0 + float(z[:t].sum())) / (t + 2.0)
        p_t = float(np.clip(p0, 1e-6, 1 - 1e-6))
        null_lik = zt * alpha + (1 - zt) * (1 - alpha)
        alt_lik = zt * p_t + (1 - zt) * (1 - p_t)
        e *= alt_lik / null_lik
        path.append(e)
        n_breach += int(zt)
        if alarm_t is None and e >= alarm_at:
            alarm_t = t
    return {
        "n": int(r.size),
        "c": float(c),
        "alpha": float(alpha),
        "breach_rate": n_breach / r.size,
        "e_final": e,
        "e_max": float(np.max(path)),
        "alarm_t": alarm_t,
        "alarmed": alarm_t is not None,
        "path": [float(v) for v in path],
    }


def tail_quota_bench(
    n: int = 800, seed: int = 0, alpha: float = 0.05, c: float = -0.02
) -> dict[str, Any]:
    """Null control + inflated-tail detection; sealed tail_quota.v1."""
    rng = np.random.default_rng(seed)
    # honest stream: sd chosen so P(N(0,sd) <= c) == alpha exactly
    sd_h = abs(c) / abs(norm.ppf(alpha))
    honest = rng.normal(0.0, sd_h, n)
    fat = rng.standard_t(3.0, n) * 0.012  # breach rate ~0.10 >> alpha
    res_h = tail_quota_eprocess(honest, c=c, alpha=alpha)
    res_f = tail_quota_eprocess(fat, c=c, alpha=alpha)
    payload: dict[str, Any] = {
        "kind": "tail_quota",
        "schema": "tail_quota.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "e-process alarms iff the breach rate at level c exceeds alpha",
            "verdict": "ok" if not res_h["alarmed"] and res_f["alarmed"] else "weak",
        },
        "interpretation": {
            "honest": {k: v for k, v in res_h.items() if k != "path"},
            "fat_tail": {k: v for k, v in res_f.items() if k != "path"},
            "note": "beta-posterior-mean alternative keeps the LR product an e-process under H0",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
