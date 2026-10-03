"""decay_watch — anytime-valid alarm that a return stream has turned negative.

The book's risk gates act on level (VaR, drawdown); this monitors *drift*:
an e-process against the composite null ``E[r_t] >= 0`` (no decay). Under
the null the process is a supermartingale, so a single alarm level
``1/alpha`` bounds the probability of *ever* falsely alarming — the right
contract for a monitor that runs forever, unlike repeated t-tests.

Construction (bounded-outcome betting e-process): each return is mapped to
``x_t = 1 - clip(r_t / bound, -1, 1) / 1`` restricted to [0, 2]... more
precisely ``x_t = clip(-r_t / bound, -1, 1) + 1`` so ``x_t in [0, 2]`` and
``E[x_t] <= 1`` exactly when ``E[r_t] >= 0``. With fixed bet size ``b``:

    e_t = prod_s (1 + b * (x_s - 1))
        = prod_s (1 - b * clip(r_s / bound, -1, 1))

which is nonnegative and a supermartingale under H0 (E[x_t] <= 1 makes each
factor's expectation <= 1). Adaptive ``b_t`` may be fitted from past data —
predictable bets keep validity; the default keeps a conservative constant.

A ``bound`` must be an honest a-priori scale for returns (e.g. a multiple
of realized vol set before monitoring begins); inflating it post-hoc to
suppress an alarm is fabricating the null, not calibrating it.

This is a risk-monitoring device, not a performance headline: it emits
the e-value path and alarm events, never P&L or Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = npt.NDArray[np.float64]

DEFAULT_ALPHA = 0.05
DEFAULT_BET = 0.5
MAX_ADAPT_BET = 0.9
MIN_ADAPT_OBS = 8


def decay_eprocess(
    returns: Array,
    bound: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    bet: float | None = DEFAULT_BET,
    adaptive: bool = True,
) -> dict[str, Any]:
    """E-process for H0: ``E[r_t] >= 0`` vs decay (negative drift).

    ``bound`` is the return scale: ``|r_t| <= bound`` is expected; larger
    magnitudes are clipped, which keeps the null valid (clipping shrinks
    x_t toward 1, i.e. toward the null — conservative under H0) but can
    dilute power on heavy-tailed books; choose ``bound`` before watching.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite")
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound must be positive and finite")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if bet is not None and (not np.isfinite(bet) or not 0.0 < bet <= MAX_ADAPT_BET):
        raise ValueError(f"bet must be in (0, {MAX_ADAPT_BET}]")

    z = np.clip(-r / bound, -1.0, 1.0)  # >0 when return negative
    e = np.empty(r.size)
    level = 1.0 / alpha
    run = 1.0
    b0 = bet if bet is not None else DEFAULT_BET
    alarms: list[int] = []
    for t in range(r.size):
        b = b0
        if adaptive and bet is None and t >= MIN_ADAPT_OBS:
            # predictable estimate of E[z] from the past only; plug-in bet
            # ~ E[z]/E[z^2] (bounded, clipped to a safe cap)
            hist = z[:t]
            denom = float(np.mean(hist * hist))
            est = float(np.mean(hist)) / max(denom, 1e-9)
            b = float(np.clip(est, 0.0, MAX_ADAPT_BET))
        factor = 1.0 + b * z[t]
        run *= max(factor, 0.0)
        e[t] = run
        if run >= level and not alarms:
            alarms.append(int(t))
    return {
        "n": int(r.size),
        "alpha": alpha,
        "bound": bound,
        "bet": b0,
        "adaptive": adaptive,
        "e_final": float(e[-1]),
        "e_max": float(e.max()),
        "alarmed": bool(alarms),
        "first_alarm_index": alarms[0] if alarms else None,
        "alarm_level": level,
        "e_path_tail": [float(v) for v in e[-16:]],
    }


def decay_demo(seed: int = 0, n: int = 400) -> dict[str, Any]:
    """Synthetic drill: a book that drifts negative halfway through alarms;
    a stationary-zero book does not (one seed each — illustrative, not the
    alpha-rate proof, which is e-process theory + the unit tests)."""
    rng = np.random.default_rng(seed)
    decaying = np.concatenate([rng.normal(0.0, 0.01, n // 4), rng.normal(-0.006, 0.01, 3 * n // 4)])
    clean = rng.normal(0.0, 0.01, n)
    payload: dict[str, Any] = {
        "kind": "decay_watch",
        "schema": "decay_watch.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "seed": seed,
            "decaying_stream": decay_eprocess(decaying, 0.02),
            "clean_stream": decay_eprocess(clean, 0.02),
        },
        "interpretation": {
            "decaying_alarms": "the negative-drift leg triggers the anytime-valid alarm",
            "clean_silent": "the zero-drift stream stays below 1/alpha",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BET",
    "decay_demo",
    "decay_eprocess",
]
