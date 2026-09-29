"""Online multiple-testing — alpha-investing over the receipt stream.

Batch BH (``corpus_inference``) answers "which committed claims survive"
at one fixed corpus. But receipts accumulate over time: each new artifact
is another hypothesis. Pool-and-rerun after every arrival silently
multiplies the family size; the honest stream procedure is
alpha-investing (Foster & Stine 2008; Javanmard & Montanari 2018 LORD):
a bounded wealth budget W is allocated per test, spent on failures, and
replenished by rejections, keeping mFDR <= alpha at every arrival time.

Rule implemented (Foster–Stine alpha-investing):

- ``alpha_t = gamma_{j_t} * W_t`` where ``gamma`` is a summable sequence
  (``gamma_j ∝ 1/(j+1)^2``, normalized to sum 1) and ``j_t`` counts tests
  since the last rejection — the longer a dry streak, the less wealth
  each new test may wager.
- Rejecting ``p_t <= alpha_t`` pays ``omega`` into wealth; a non-rejection
  forfeits ``psi_t = alpha_t / (1 - alpha_t)`` (the Foster–Stine cost
  function — paying only alpha_t would let an adversary burn wealth
  sublinearly on near-boundary losses).
- Wealth floors at zero (alpha_t = 0 ⇒ nothing can be wagered — a
  dead lane stays dead rather than borrowing future wealth).

Deliberately conservative: p-values are validated per update, the
gamma index restarts only on true rejections, and the verdict object
records every wager so a stream can be replayed for audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _gamma(j: int) -> float:
    """Summable spending sequence: gamma_j ∝ 1/(j+1)^2, Σ = 1."""
    zeta2 = math.pi**2 / 6.0
    return (1.0 / (j + 1) ** 2) / zeta2


@dataclass(frozen=True)
class OnlineTestState:
    """State after one stream update."""

    index: int
    p_value: float
    alpha_t: float
    rejected: bool
    wealth: float


@dataclass
class OnlineFDR:
    """Foster–Stine alpha-investing controller over a p-value stream.

    ``level`` is the target mFDR bound; ``initial_wealth`` and ``payout``
    default to the standard split ``w0·α`` / ``(1−w0)·α`` with w0=0.5.
    """

    level: float = 0.05
    initial_wealth: float | None = None
    payout: float | None = None
    _w0_fraction: float = 0.5
    _wealth: float = field(init=False)
    _payout_resolved: float = field(init=False)
    _since_rejection: int = field(default=0, init=False)
    _states: list[OnlineTestState] = field(default_factory=list, init=False)
    _n_rejections: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must lie in (0, 1)")
        if self.initial_wealth is None:
            self.initial_wealth = self._w0_fraction * self.level
        if self.payout is None:
            self.payout = (1.0 - self._w0_fraction) * self.level
        if not (0.0 < self.initial_wealth <= self.level):
            raise ValueError("initial_wealth must lie in (0, level]")
        if not (self.payout > 0.0):
            raise ValueError("payout must be positive")
        self._wealth = float(self.initial_wealth)
        self._payout_resolved = float(self.payout)

    def update(self, p_value: float) -> OnlineTestState:
        """Test one new p-value; returns the post-update state."""
        p = float(p_value)
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError("p_value must lie in [0, 1]")
        j = self._since_rejection
        alpha_t = min(_gamma(j) * self._wealth, self._wealth)
        rejected = p <= alpha_t
        if rejected:
            self._wealth = self._wealth + self._payout_resolved
            self._n_rejections += 1
            self._since_rejection = 0
        else:
            self._wealth = max(
                0.0, self._wealth - alpha_t / (1.0 - alpha_t) if alpha_t < 1.0 else 0.0
            )
            self._since_rejection += 1
        state = OnlineTestState(
            index=len(self._states),
            p_value=p,
            alpha_t=alpha_t,
            rejected=rejected,
            wealth=self._wealth,
        )
        self._states.append(state)
        return state

    @property
    def wealth(self) -> float:
        return self._wealth

    @property
    def rejections(self) -> list[int]:
        return [s.index for s in self._states if s.rejected]

    def stream_report(self) -> dict[str, Any]:
        return {
            "kind": "online_fdr.v1",
            "n_tests": len(self._states),
            "n_rejections": self._n_rejections,
            "rejection_indices": self.rejections,
            "final_wealth": self._wealth,
            "level": self.level,
            "evidence": ["foster_stine_alpha_investing", "mfdr_bounded", "summable_gamma"],
        }
