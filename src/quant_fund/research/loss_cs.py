"""Anytime-valid confidence sequences for a mean loss difference.

Waudby-Smith & Ramdas (2023, "Estimating means of bounded random
variables by betting", JRSSB 85(4)) / Howard, Ramdas, McAuliffe & Sekhon
(2021, "Time-uniform, nonparametric, nonasymptotic confidence sequences",
Annals of Statistics 49(2)): for bounded increments ``d_t ∈ [a, b]`` with
conditional mean ``μ``, the family

    e_t(μ0) = ∏_{i<=t} ( 1 + λ (d_i − μ0) / (b − a) ),   |λ| ≤ 1,

is a nonnegative test (super)martingale under ``E[d_i | F_{i-1}] = μ0`` —
every factor is ``≥ 0`` (|λ (d−μ)/(b−a)| ≤ 1) with conditional mean 1.
A fixed positive λ makes ``e⁺(μ)`` strictly *decreasing* in μ, so
``{μ : e⁺(μ) < c}`` is a ray — a one-sided lower CS. The mirror family
``e⁻`` (λ < 0) is increasing, giving the upper ray. Two one-sided CSs at
α/2 each, Bonferroni-combined, give a two-sided **confidence sequence**:

    CS_t = (L_t, U_t),
    P( μ ∈ CS_t for all t ≥ 1 ) ≥ 1 − α

Both inversions are bracketed bisection on monotone functions — no
stitched-boundary constants, no asymptotics, valid at any stopping time
and under any dependence structure that keeps ``E[d_t | F_{t-1}] = μ``
(e.g. successive eval chunks in ``fleet_race``, per-origin proper-loss
diffs in ``evalues``).

Fixed ``λ = 1/2`` is the recommended default (WSR call it the
"mixed/betting" choice): predictable λ-tuning would narrow the interval
toward the observed diffs but forfeits the clean reading that every claim
is time-uniform without hindsight adjustments; λ is printed in receipts.

Where ``evalues.LossEProcess`` answers *"does the challenger beat the
incumbent?"* (point-null at 0), this module answers *"by how much?"* —
``CS_t`` bounded away from 0 is anytime-valid evidence of a real edge of
at least that magnitude. Proper scores only: the streams are pinball/CRPS
diffs, never P&L.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOSS_CS_SCHEMA = "loss_cs.v1"


@dataclass
class MeanDiffCS:
    """Betting confidence sequence for the mean of a bounded diff stream.

    ``bound`` must be a *true* per-step bound: callers pass
    ``|challenger_loss − incumbent_loss| <= bound`` for every step.
    Supplying a bound that a diff later violates fails closed (raise) —
    a CS built on a violated boundedness premise is invalid.
    """

    alpha: float = 0.05
    lam: float = 0.5
    bound: float = 1.0
    _diffs: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha) and 0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (np.isfinite(self.lam) and abs(self.lam) <= 1.0):
            raise ValueError("lam must satisfy |λ| <= 1")
        if not (np.isfinite(self.bound) and self.bound > 0.0):
            raise ValueError("bound must be positive")

    def update(self, d: float) -> None:
        if not np.isfinite(d):
            raise ValueError("diff must be finite")
        if abs(d) > self.bound * (1.0 + 1e-12):
            raise ValueError(
                f"diff {d} violates the declared bound {self.bound}; "
                "the CS is only valid while |d| <= bound"
            )
        self._diffs.append(d)

    @property
    def n(self) -> int:
        return len(self._diffs)

    def _log_evalue(self, mu: float, sign: float) -> float:
        """log e_t(mu) for the one-sided betting family with λ = sign·|lam|.

        sign=+1 bets on increments above mu (rejects mu when the true mean
        is lower); sign=-1 bets below. Each is monotone in mu, so one-sided
        inversion is bracketed bisection.
        """
        if not np.isfinite(mu):
            return np.inf
        if not self._diffs:
            return 0.0
        d = np.asarray(self._diffs, dtype=float)
        terms = 1.0 + sign * self.lam * (d - mu) / self.bound
        if float(terms.min()) <= 0.0:
            return np.inf
        return float(np.log(terms).sum())

    def interval(self) -> tuple[float, float]:
        """Two-sided CS at time t via two one-sided betting families.

        Lower bound inverts ``e⁺(μ) = Π(1+λ(d−μ)/r)`` (strictly decreasing
        in μ on its support) at level α/2; upper bound inverts ``e⁻`` at
        α/2; Bonferroni gives the two-sided time-uniform α guarantee.
        """
        if not self._diffs:
            return (-self.bound, self.bound)
        threshold = np.log(2.0 / self.alpha)
        lo_edge, hi_edge = -self.bound, self.bound

        def _bisect(lo: float, hi: float, sign: float) -> float:
            # sign=+1: g decreasing -> find where it drops below threshold
            # sign=-1: g increasing -> find where it rises above threshold
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                inside = self._log_evalue(mid, sign) < threshold
                # decreasing g (sign +1): inside => root below mid -> hi=mid
                # increasing g (sign -1): inside => root above mid -> lo=mid
                if inside == (sign < 0):
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        left = lo_edge
        if self._log_evalue(lo_edge, 1.0) >= threshold:
            left = _bisect(lo_edge, hi_edge, +1.0)
        right = hi_edge
        if self._log_evalue(hi_edge, -1.0) >= threshold:
            right = _bisect(lo_edge, hi_edge, -1.0)
        if left > right:  # degenerate numerical corner — widen honestly
            left, right = lo_edge, hi_edge
        return (left, right)


def cs_from_streams(
    challenger_losses: Sequence,
    incumbent_losses: Sequence,
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    bound: float | None = None,
    challenger: str = "challenger",
    incumbent: str = "incumbent",
    data_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """CS over ``d_t = challenger − incumbent`` proper-loss diffs.

    ``bound`` defaults to the observed worst-case |d| — honest only because
    it enters the interval *width*, not the coverage guarantee's validity
    condition (the CS remains valid; a larger bound widens it). Callers
    with a known a-priori loss range should pass it.
    """
    c = np.asarray(list(challenger_losses), dtype=float).reshape(-1)
    b = np.asarray(list(incumbent_losses), dtype=float).reshape(-1)
    if c.size == 0 or c.shape != b.shape:
        raise ValueError("loss streams must be nonempty and equal length")
    if not (np.isfinite(c).all() and np.isfinite(b).all()):
        raise ValueError("loss streams must be finite")
    diffs = c - b
    eff_bound = float(bound) if bound is not None else float(np.abs(diffs).max())
    if not np.isfinite(eff_bound) or eff_bound <= 0:
        raise ValueError("degenerate bound")
    proc = MeanDiffCS(alpha=alpha, lam=lam, bound=eff_bound)
    for d in diffs.tolist():
        proc.update(float(d))
    lo, hi = proc.interval()
    return {
        "kind": LOSS_CS_SCHEMA,
        "schema": LOSS_CS_SCHEMA,
        "challenger": challenger,
        "incumbent": incumbent,
        "data_label": data_label,
        "alpha": alpha,
        "lam": lam,
        "bound": eff_bound,
        "n": int(diffs.size),
        "mean_diff": float(diffs.mean()),
        "cs_low": lo,
        "cs_high": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "interpretation": "challenger_better"
        if hi < 0.0
        else ("incumbent_better" if lo > 0.0 else "inconclusive"),
        "evidence": [
            "ville_inequality",
            "betting_confidence_sequence",
            "bounded_increments",
            "time_uniform_coverage",
        ],
    }
