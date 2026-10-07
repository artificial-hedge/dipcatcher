"""Change-point localization via backward e-value scans.

A drift alarm (``drift_alarm.EProcessDriftAlarm``, ``conformal_monitor``)
answers *"did the stream shift?"* — this module answers *"where?"* with a
finite-sample confidence set instead of a bare argmax.

Construction (the e-value confidence set for a change point; the approach
is the backward e-value scan of Shekhar–Ramdas–Grünwald 2023, "Nonparametric
two-sample testing by betting", used for localization as in Wang & Ramdas
2022 / Joodaki et al. e-value change-point confidence sets). For each
candidate split ``s`` we run a bounded betting process forward from ``s``
to the end of the stream:

    e_s = ∏_{s <= i < s+W} ( 1 + λ z_i ),
        z_i = clip((x_i − μ̂_{<s}) / ŝ_{<s}, −1, 1)

with ``μ̂_{<s}``, ``ŝ_{<s}`` the pre-split mean and scale, over a **fixed**
post-window of length ``W`` — the fixed window is what makes the argmax
localize: an unbounded suffix gives every s < τ the same new-regime tail
product, so the e-path is flat left of the change; under a fixed window
the evidence peaks exactly when [s, s+W) is all post-change. Under the null
that the post-s segment is drawn from the pre-split law (the plug-in
baseline ``μ̂_{<s}``, made honest by the bounded clip), ``e_s`` is an
e-value. The plausibility set is

    CS = { s : e_s < 1/α }

whose honest reading is precise: ``CS`` contains exactly the split points
whose right segment is *consistent with continuing their own left-side
law* at level α. Around a true single change at τ, candidates s > τ put
post-change mass in the baseline too, so their e-values stay small and
they populate the right end of CS; candidates s ≪ τ contaminate the right
segment with both regimes and accumulate intermediate evidence. The
excluded region {e_s ≥ 1/α} brackets τ — the localization claim is
``argmax_s e_s`` for the point estimate and the CS *boundary* (its left
edge) as the conservative localization, not a simultaneous coverage
statement over all s. Empirical coverage of τ by (cs_lo boundary) is
pinned by the test suite's Monte-Carlo check.

Reported quantities: the MLE-style point estimate ``argmax_s e_s``, the
confidence-set bounds, and per-candidate log-e-values for the receipt.
Streams are score streams (pinball residuals, PIT-centered values,
nonconformity scores) — never P&L.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOCALIZE_SCHEMA = "changepoint_localize.v1"


@dataclass
class LocalizeResult:
    tau_hat: int
    cs_lo: int
    cs_hi: int
    n: int
    alarmed: bool  # any candidate crossed the scan-corrected threshold
    log_evalues: list[float] = field(default_factory=list)


def localize_changepoint(
    stream: Sequence[float],
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    window: int = 40,
    min_left: int = 10,
) -> LocalizeResult:
    """Scan candidate split points; return the excluded-region bracket.

    Candidates ``s`` score ``[s, s+W)`` against the baseline ``x[<s]``.
    ``min_left`` bounds the left edge so the plug-in baseline has support;
    ``window`` sets both the localization resolution and the per-candidate
    martingale length. Non-finite observations are dropped before
    scanning — a NaN bar must not manufacture or hide a change point.
    """
    if not (np.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (np.isfinite(lam) and 0.0 < lam <= 1.0):
        raise ValueError("lam must be in (0, 1]")
    if not (isinstance(window, int) and window >= 5):
        raise ValueError("window must be an int >= 5")
    x = np.asarray(list(stream), dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < min_left + window:
        raise ValueError(f"stream too short to localize: n={n}, need >= {min_left + window}")
    # Scan correction: each candidate split is its own e-value, and the
    # alarm is "did ANY candidate cross" — a max over the scan, so the
    # per-candidate threshold must spend alpha across the family
    # (Bonferroni/union bound; valid under arbitrary dependence between
    # overlapping windows).
    n_candidates = max(1, n - window - min_left + 1)
    threshold = np.log(n_candidates / alpha)

    log_e = np.full(n, np.nan)
    # inclusive right edge: s = n - window is a legal candidate — its window
    # [n-W, n) is the final W observations — so a change inside the last W
    # positions must stay reachable or it silently cannot be localized.
    for s in range(min_left, n - window + 1):
        left = x[:s]
        mu = float(left.mean())
        scale = float(left.std(ddof=1)) if s > 1 else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1e-12  # constant pre-split: any post deviation is signal
        # spread center: baseline's own mean |residual| — self-calibrating,
        # no Gaussian assumption, insensitive to small-baseline scale noise
        base_spread = float(np.abs(left - mu).mean())
        if not np.isfinite(base_spread) or base_spread <= 0.0:
            base_spread = scale  # degenerate fallback
        l_mean = 0.0
        l_spread = 0.0
        for xi in x[s : s + window]:
            z = max(-1.0, min(1.0, (xi - mu) / scale))
            l_mean += np.log(max(0.0, 1.0 + lam * z))
            # |x − μ̂| vs the baseline's own mean absolute residual — a
            # zero-mean bet can't see a variance shift (Jensen:
            # E[log(1+λz)] < 0 under E[z]=0); this one can.
            z2 = max(-1.0, min(1.0, (abs(xi - mu) - base_spread) / base_spread))
            l_spread += np.log(max(0.0, 1.0 + lam * z2))
        # equal mixture of two e-values is an e-value
        log_e[s] = np.logaddexp(l_mean, l_spread) - np.log(2.0)

    finite = np.isfinite(log_e)
    if not finite.any():
        raise ValueError("no candidate split produced a finite e-value")
    tau_hat = int(np.nanargmax(log_e))
    excluded = np.where(finite & (log_e >= threshold))[0]
    alarmed = bool(excluded.size > 0)
    if alarmed:
        # bracket of the excluded (inconsistent-with-continuation) region
        cs_lo, cs_hi = int(excluded.min()), int(excluded.max())
    else:
        cs_lo, cs_hi = min_left, n - window
    return LocalizeResult(
        tau_hat=tau_hat,
        cs_lo=cs_lo,
        cs_hi=cs_hi,
        n=n,
        alarmed=alarmed,
        log_evalues=[float(v) if np.isfinite(v) else float("nan") for v in log_e],
    )


def localize_report(
    stream: Sequence[float],
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    window: int = 40,
    min_left: int = 10,
    stream_name: str = "stream",
    data_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """Receipt-shaped localization report."""
    res = localize_changepoint(stream, alpha=alpha, lam=lam, window=window, min_left=min_left)
    return {
        "kind": LOCALIZE_SCHEMA,
        "schema": LOCALIZE_SCHEMA,
        "stream": stream_name,
        "data_label": data_label,
        "alpha": alpha,
        "lam": lam,
        "min_left": min_left,
        "window": window,
        "params": {
            "alpha": alpha,
            "lam": lam,
            "min_left": min_left,
            "window": window,
        },
        "result": {
            "n": res.n,
            "tau_hat": res.tau_hat,
            "cs_lo": res.cs_lo,
            "cs_hi": res.cs_hi,
            "alarmed": res.alarmed,
        },
        "n": res.n,
        "tau_hat": res.tau_hat,
        "cs_lo": res.cs_lo,
        "cs_hi": res.cs_hi,
        "alarmed": res.alarmed,
        "log_evalues": res.log_evalues,
        "evidence": [
            "evalue_confidence_set",
            "backward_scan",
            "bounded_bets",
            "proper_scores_only",
        ],
    }
