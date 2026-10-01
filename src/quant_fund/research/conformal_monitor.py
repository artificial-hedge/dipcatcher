"""Conformal-martingale distribution monitor (Vovk's change detection).

A complementary detector to ``drift_alarm``: where the e-process there
bets on a *level shift* in the mean, this module is **nonparametric** —
it alarms on *any* departure from exchangeability of the stream's
nonconformity scores (variance change, tail thickening, serial
dependence, distributional breaks a mean-shift test is blind to).

Machinery (Vovk–Nouretdinov–Gammerman, "Testing exchangeability
on-line"; also the conformal martingale strand of Vovk's 2021
"Testing randomness online"):

1. Each new observation ``x_t`` is scored against the trailing
   calibration window by a nonconformity measure — absolute deviation
   from the window's running median, i.e. ``s_t = |x_t − m_{<t}|``.
2. ``p_t = (1 + #{i ∈ window: s_i ≥ s_t}) / (window + 1)`` — the
   conformal p-value; exactly valid (super-uniform) under
   exchangeability, no distributional assumption.
3. The p-stream is fed to a **κ-martingale** ``M = Π κ·p_i^{κ−1}``
   (``κ ∈ (0,1)`` — power betting against uniformity): under the null
   ``E[M] ≤ 1`` at every stopping time, so an alarm at ``M ≥ 1/α`` has
   P(ever fire) ≤ α by Ville — identical stopping validity to the
   betting e-process, a strictly different test statistic.

Fails closed: non-finite observations consume a step and emit
``inconclusive`` — never silently reset or skipped (a skipped NaN is a
laundered observation).

Two modes, deliberately different validity guarantees:

- ``mode="fixed"`` (default): the calibration bag is the first
  ``window`` observations, never updated. Exchangeability w.r.t. the
  fixed bag holds *exactly* under the null, so the Ville bound is
  exact — and a persistent regime change keeps producing small p's
  forever (no re-conformation): the detector for "has this head
  degraded since calibration".
- ``mode="adaptive"``: the bag is the trailing ``window`` scores.
  Bounded memory, and the detector catches *episodes* (a burst of
  nonconforming points) — but a persistent shift re-conforms after the
  window drains, and overlapping bags correlate the p-stream so the
  bound is asymptotic, not exact. Stated, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONFORMAL_MONITOR_SCHEMA = "conformal_monitor.v1"


@dataclass
class ConformalMartingale:
    """Conformal martingale over a sliding calibration window.

    ``window``: calibration bag size (first N observations in fixed
    mode; trailing N in adaptive). ``kappa``: betting exponent in
    (0, 1) — smaller bets harder on small p-values (0.5 = sqrt rule).
    ``mode``: "fixed" (exact validity, persistent-shift power) or
    "adaptive" (bounded memory, episode detection).
    """

    alpha: float = 0.05
    kappa: float = 0.5
    window: int = 50
    init_scale: float = 1e-6
    mode: str = "fixed"

    _bag: list[float] = field(default_factory=list)  # calibration scores
    _xs: list[float] = field(default_factory=list)  # raw stream
    _log_m: float = 0.0
    _alarm_index: int | None = None
    _pvals: list[float] = field(default_factory=list)
    _n_inconclusive: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (0.0 < self.kappa < 1.0):
            raise ValueError("kappa must be in (0, 1)")
        if self.window < 5:
            raise ValueError("window must be >= 5")
        if self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("mode must be 'fixed' or 'adaptive'")

    def _score(self, x: float) -> float:
        """Nonconformity: |x − median of the calibration xs|."""
        w = self._xs[-self.window :] if self.mode == "adaptive" else self._xs[: self.window]
        if not w:
            return 0.0
        return abs(x - float(np.median(w)))

    def _pvalue(self, s: float) -> float:
        """Conformal p of score s against the calibration bag."""
        if not self._bag:
            return 1.0
        greater_equal = sum(1 for b in self._bag if b >= s)
        return float((1.0 + greater_equal) / (len(self._bag) + 1.0))

    def update(self, x: float) -> tuple[float, float]:
        """Observe x → (conformal p_t, martingale M_t)."""
        v = float(x)
        if not np.isfinite(v):
            self._n_inconclusive += 1
            return float("nan"), float(np.exp(self._log_m))
        self._xs.append(v)
        s = self._score(v)
        p = self._pvalue(s)
        # Clamp p away from exact 0/1 so the κ-power bet stays finite —
        # p ∈ [1/(m+1), 1] already, but guard anyway.
        p = float(min(max(p, 1.0 / (self.window + 1.0)), 1.0))
        self._pvals.append(p)
        # κ-martingale factor: κ·p^{κ−1} (E ≤ 1 under uniform p)
        e_mult = self.kappa * p ** (self.kappa - 1.0)
        self._log_m += float(np.log(max(e_mult, 1e-300)))
        if self.mode == "adaptive" or len(self._bag) < self.window:
            self._bag.append(s)
        if len(self._bag) > self.window:
            self._bag.pop(0)
        m = float(np.exp(self._log_m))
        if self._alarm_index is None and m >= 1.0 / self.alpha:
            self._alarm_index = len(self._pvals) - 1
        return p, m

    @property
    def alarmed(self) -> bool:
        return self._alarm_index is not None

    @property
    def alarm_index(self) -> int | None:
        return self._alarm_index

    @property
    def martingale(self) -> float:
        return float(np.exp(self._log_m))


def monitor(
    stream: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    kappa: float = 0.5,
    window: int = 50,
    mode: str = "fixed",
    data_label: str = "UNKNOWN",
) -> dict[str, object]:
    """Run the conformal martingale over a stream → conformal_monitor.v1."""
    cm = ConformalMartingale(alpha=alpha, kappa=kappa, window=window, mode=mode)
    last_m = 1.0
    for x in np.asarray(stream, dtype=float).ravel():
        _, last_m = cm.update(float(x))
    return {
        "kind": CONFORMAL_MONITOR_SCHEMA,
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
        "n_obs": len(cm._pvals),
        "alpha": alpha,
        "kappa": kappa,
        "window": window,
        "mode": mode,
        "alarmed": cm.alarmed,
        "alarm_index": cm.alarm_index,
        "final_martingale": last_m,
        "n_inconclusive": cm._n_inconclusive,
        "evidence": [
            "conformal_pvalues_exchangeability",
            "kappa_martingale_stopping_valid",
            "nonparametric_distribution_shift",
        ],
    }
