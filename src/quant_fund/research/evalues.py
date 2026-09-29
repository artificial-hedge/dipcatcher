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
  bet clipped to ``[0, lam]``. The default ``lambda_policy="kelly"``
  plug-in turns the running challenger win-rate ``p_hat`` into
  ``lam_i = clip(2*p_hat - 1, 0, lam)`` (strict history only; ``lam``
  is the aggressiveness cap, not a fixed bet). The alternative
  ``lambda_policy="online"`` replaces the flat-average plug-in with
  GRAPA-style exponential-gradient ascent on realized log-growth —
  a constant-step tracker that adapts within a bounded window instead
  of decaying at rate ``1/n``, which is where power is recovered under
  serial dependence and regime change. Under the null ``lam_i >= 0``
  is all that validity needs.
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


@dataclass
class OnlineLambda:
    """Predictable adaptive bet via exponential-gradient ascent on log-growth.

    GRAPA-style online bet selection (Waudby-Smith & Ramdas 2024, the
    gradient-ascent instantiation): on the sign channel the factor is
    ``e_i = 1 + lam_i * x_i`` with the win outcome ``x_i = -sign(d_i)``
    (+1 when the challenger is better). For binary ``x`` the
    growth-optimal fixed bet is ``lam* = E[x]`` — the Kelly plug-in
    estimates it with a flat ``1/n`` running mean, which dilutes under
    regime change. This policy instead runs constant-step-size
    stochastic approximation on the realized log-growth gradient
    ``grad_i = x_i / (1 + lam_i * x_i)`` — the derivative of
    ``log(1 + lam * x_i)`` — smoothed by an EMA before each step:

    - ``predict()`` returns ``lam_t`` before origin ``t`` is seen — a
      deterministic function of ``x_1..x_{t-1}`` only (predictable).
    - ``observe(x_i)`` folds the realized gradient into the EMA and
      steps ``lam <- clip(lam + eta * grad_hat, 0, lam_max)``. The
      fixed point solves ``E[x/(1+lam*x)] = 0`` — i.e. ``lam* = E[x]``
      for the sign channel — while the constant step keeps the
      estimator non-forgetting so it tracks time-varying edge.

    Ville validity is preserved regardless of adaptivity: ``lam_t`` is
    measurable w.r.t. ``F_{t-1}`` by construction, so under the median
    null ``E[x_t | F_{t-1}] <= 0`` implies
    ``E[e_t | F_{t-1}] = 1 + lam_t * E[x_t | F_{t-1}] <= 1`` — every
    factor is still a valid e-value and the product a nonnegative
    supermartingale. Adaptivity trades only power inside the valid
    envelope. The instance is mutable state: a ``LossEProcess`` owning
    it advances it one observation per origin.
    """

    lam_max: float
    eta: float = 0.2
    smooth: float = 0.3
    _lam: float = 0.0
    _grad_ema: float = 0.0
    _seen: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.lam_max < 1.0):
            raise ValueError("lam_max must lie in (0, 1)")
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be positive and finite")
        if not (0.0 < self.smooth <= 1.0):
            raise ValueError("smooth must lie in (0, 1]")

    def predict(self) -> float:
        """Bet for the next origin; uses only observed rounds (strict history)."""
        return self._lam

    def observe(self, x: float) -> None:
        """Fold one realized sign-win ``x in [-1, 1]`` into the bet state."""
        grad = x / (1.0 + self._lam * x)
        self._seen += 1
        self._grad_ema += self.smooth * (grad - self._grad_ema)
        # Bias-corrected EMA of the realized gradient — the step target.
        g_hat = self._grad_ema / (1.0 - (1.0 - self.smooth) ** self._seen)
        self._lam = float(np.clip(self._lam + self.eta * g_hat, 0.0, self.lam_max))


@dataclass(frozen=True)
class EProcessState:
    """Snapshot of the promotion process at one origin."""

    origin: int
    evalue: float
    anytime_p: float
    promoted: bool
    lam: float = 0.0


@dataclass
class LossEProcess:
    """Sequential e-process promoting a challenger when its losses beat the incumbent.

    ``challenger_losses`` / ``incumbent_losses`` are per-origin proper
    scores of identical length (lower = better). All state updates are
    causal: the bet fraction at origin ``i`` uses only ``d_j`` for
    ``j < i``, so appending future observations can never rewrite a
    reported state.

    ``lambda_policy`` selects the predictable bet: ``"kelly"`` (default)
    keeps the running-win-rate plug-in, ``"online"`` builds an
    :class:`OnlineLambda` bounded by ``lam`` and tuned by
    ``online_eta``/``online_smooth``, and an :class:`OnlineLambda`
    instance may be passed directly for custom tuning. Every policy is
    strictly predictable and bounded, so the Ville guarantee is
    unchanged; the default is a no-op for existing callers.
    """

    lam: float = 0.5
    alpha: float = 0.05
    init_scale: float = 1e-3
    lambda_policy: str | OnlineLambda = "kelly"
    online_eta: float = 0.2
    online_smooth: float = 0.3
    _states: list[EProcessState] = field(default_factory=list)
    _diffs: list[float] = field(default_factory=list)
    _wins: int = 0
    _log_e: float = 0.0
    _promotion_origin: int | None = None
    _online: OnlineLambda | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must lie in (0, 1)")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not np.isfinite(self.init_scale) or self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive and finite")
        policy = self.lambda_policy
        if isinstance(policy, OnlineLambda):
            self._online = policy
        elif policy == "online":
            self._online = OnlineLambda(
                lam_max=self.lam, eta=self.online_eta, smooth=self.online_smooth
            )
        elif policy != "kelly":
            raise ValueError("lambda_policy must be 'kelly', 'online', or an OnlineLambda")

    def _predictable_lam(self) -> float:
        """Bet for the current origin from strict history; in [0, lam]."""
        if self._online is not None:
            return self._online.predict()
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
        lam_i = self._predictable_lam()
        e = 1.0 - lam_i * g
        # e in (1-lam, 1+lam) — strictly positive by construction.
        if self._online is not None:
            self._online.observe(-g)  # win outcome +1 when challenger better
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
            lam=lam_i,
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
    lambda_policy: str = "kelly",
    online_eta: float = 0.2,
    online_smooth: float = 0.3,
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
    proc = LossEProcess(
        lam=lam,
        alpha=alpha,
        lambda_policy=lambda_policy,
        online_eta=online_eta,
        online_smooth=online_smooth,
    )
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
        "lambda_policy": lambda_policy,
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
