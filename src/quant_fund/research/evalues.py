"""Anytime-valid head promotion — e-processes over per-origin score gaps.

The fleet's significance gates are batch procedures: DM/MCS/PBO answer
"was the challenger better over this fixed horizon" after all origins
are scored. This lane answers the sequential question — *at which origin
may we promote the challenger and stop early* — with a test martingale
valid at arbitrary stopping times.

Construction (betting e-value, Waudby-Smith & Ramdas 2024, "Estimating
means of bounded random variables by betting"; Ville 1939):

- Per origin i, the loss differential ``d_i = challenger_i - incumbent_i``
  (positive = incumbent better).
- The bet is on the *sign* of ``d_i``: ``g_i = sign(d_i)`` in {-1, 0, +1}.
  Magnitude is deliberately ignored — a distribution-free construction.
- E-factor ``e_i = 1 - lam_i * g_i`` with ``lam_i`` a *predictable*
  Kelly-style plug-in clipped to ``[0, lam]``: the running challenger
  win-rate ``p_hat`` turns into ``lam_i = clip(2*p_hat - 1, 0, lam)``
  (strict history only; ``lam`` is the aggressiveness cap, not a fixed
  bet). Under the null ``lam_i >= 0`` is all that validity needs.
- Under H0 — challenger does not beat the incumbent on the typical
  origin, ``P(d_i < 0 | F_{i-1}) <= P(d_i > 0 | F_{i-1})`` (for continuous
  diffs, ``median(d_i | F_{i-1}) >= 0``) — ``E[sign(d_i) | F_{i-1}] >= 0``,
  so ``E[e_i | F_{i-1}] <= 1`` and ``E_t = prod e_i`` is a nonnegative
  supermartingale starting at 1. The null is *conditional*: any iid
  stream with ``median(d) >= 0`` qualifies regardless of tails, skew,
  or scale, but a stream whose sign is predictable from the past (e.g.
  strong mean-reversion) is outside it.

By Ville's inequality, ``p_t = min(1, 1 / E_t)`` is an anytime-valid
p-value and promotion at the first origin where ``E_t >= 1/alpha``
controls the false-promotion rate at ``alpha`` under arbitrary stopping.

Why the sign bet: no e-process can test the raw mean-null ``E[d] >= 0``
for unbounded differentials — a nonnegative factor must satisfy
``e(x) <= 1`` at every ``x > 0`` (a point mass at ``x`` is a null), and
the same argument forces ``e <= 1`` everywhere, leaving no power. Any
bounded magnitude bet ``clip(d/s, -1, 1)`` saturates to a sign test but
claims a mean-null it cannot honor: under a mean-0 skewed stream the
clipped bet's own mean turns negative, so ``E[e_i] > 1`` and the
supermartingale is lost (observed empirically: centered-gamma and
two-point mean-0 streams promote ~50-100% of the time). The sign bet
tests the strongest distribution-free notion of "no better" — the
median — and keeps ``E[e_i] <= 1`` under arbitrary tails, skew, and
misspecification; ``lam_i`` adaptivity only trades power inside the
valid envelope, never validity. ``init_scale`` is retained for API
compatibility and diagnostics — the sign bet needs no scale.

Deliberately conservative: every factor lies in ``(1 - lam, 1 + lam)``,
strictly positive, so no finite sample can zero the martingale outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_PRIOR_WINS = 1.0  # Laplace pseudo-count: p_hat shrinks toward 0.5 early


@dataclass(frozen=True)
class EProcessState:
    """Snapshot of the promotion process at one origin."""

    origin: int
    evalue: float
    anytime_p: float
    promoted: bool


@dataclass
class LossEProcess:
    """Sequential e-process promoting a challenger when its losses beat the incumbent.

    ``challenger_losses`` / ``incumbent_losses`` are per-origin proper
    scores of identical length (lower = better). All state updates are
    causal: the bet fraction at origin ``i`` uses only ``d_j`` for
    ``j < i``, so appending future observations can never rewrite a
    reported state.
    """

    lam: float = 0.5
    alpha: float = 0.05
    init_scale: float = 1e-3
    _states: list[EProcessState] = field(default_factory=list)
    _diffs: list[float] = field(default_factory=list)
    _wins: int = 0
    _log_e: float = 0.0
    _promotion_origin: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must lie in (0, 1)")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not np.isfinite(self.init_scale) or self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive and finite")

    def _predictable_lam(self) -> float:
        """Kelly plug-in on the sign channel from strict history; in [0, lam]."""
        n = len(self._diffs)
        p_hat = (self._wins + _PRIOR_WINS) / (n + 2.0 * _PRIOR_WINS)
        return min(max(2.0 * p_hat - 1.0, 0.0), self.lam)

    def update(self, challenger_loss: float, incumbent_loss: float) -> EProcessState:
        """Append one origin's losses and return the new state."""
        c = float(challenger_loss)
        b = float(incumbent_loss)
        if not (np.isfinite(c) and np.isfinite(b)):
            raise ValueError("losses must be finite")
        d = c - b
        g = float(np.sign(d))
        e = 1.0 - self._predictable_lam() * g
        # e in (1-lam, 1+lam) — strictly positive by construction.
        self._log_e += float(np.log(e))
        self._wins += int(d < 0)
        self._diffs.append(d)
        log_cap = float(np.log(1.0 / self.alpha))
        if self._promotion_origin is None and self._log_e >= log_cap:
            self._promotion_origin = len(self._diffs) - 1
        evalue = float(np.exp(min(self._log_e, 700.0)))
        state = EProcessState(
            origin=len(self._diffs) - 1,
            evalue=evalue,
            anytime_p=float(min(1.0, 1.0 / evalue)),
            promoted=self._promotion_origin is not None,
        )
        self._states.append(state)
        return state

    @property
    def promotion_origin(self) -> int | None:
        return self._promotion_origin

    @property
    def states(self) -> list[EProcessState]:
        return list(self._states)


def promotion_report(
    challenger_losses: Array | list[float],
    incumbent_losses: Array | list[float],
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    challenger: str = "challenger",
    incumbent: str = "incumbent",
) -> dict[str, Any]:
    """Receipt-shaped promotion verdict over two per-origin loss streams."""
    c = np.asarray(list(challenger_losses), dtype=np.float64).reshape(-1)
    b = np.asarray(list(incumbent_losses), dtype=np.float64).reshape(-1)
    if c.size == 0 or c.shape != b.shape:
        raise ValueError("loss streams must be nonempty and equal length")
    if not (np.isfinite(c).all() and np.isfinite(b).all()):
        raise ValueError("loss streams must be finite")
    proc = LossEProcess(lam=lam, alpha=alpha)
    for ci, bi in zip(c.tolist(), b.tolist(), strict=True):
        proc.update(ci, bi)
    final = proc.states[-1]
    diffs = c - b
    return {
        "kind": "evalue_promotion.v1",
        "challenger": challenger,
        "incumbent": incumbent,
        "alpha": alpha,
        "lam": lam,
        "n_origins": int(c.size),
        "final_evalue": final.evalue,
        "anytime_p": final.anytime_p,
        "promotion_origin": proc.promotion_origin,
        "promoted": proc.promotion_origin is not None,
        "mean_loss_diff": float(np.mean(diffs)),
        "challenger_win_rate": float(np.mean(diffs < 0)),
        "scale_median": float(np.median(np.abs(diffs))),
        "evidence": [
            "ville_inequality",
            "nonnegative_test_martingale",
            "predictable_bet",
            "median_null",
            "stopping_time_valid",
        ],
    }
