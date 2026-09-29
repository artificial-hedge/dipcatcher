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
- Normalized bet ``g_i = clip(d_i / scale_i, -1, 1)`` where ``scale_i`` is
  a *predictable* robust scale — the running median of |d| over strict
  history (zero-origin bootstrap by ``init_scale``); nothing reads the
  current observation.
- E-factor ``e_i = 1 - lam * g_i`` with fixed ``lam`` in (0, 1). Under
  H0 — challenger is no better, E[d_i | F_{i-1}] >= 0 — monotonicity of
  ``g`` gives E[e_i | F_{i-1}] <= 1, so ``E_t = prod e_i`` is a
  nonnegative supermartingale starting at 1.

By Ville's inequality, ``p_t = min(1, 1 / E_t)`` is an anytime-valid
p-value and promotion at the first origin where ``E_t >= 1/alpha``
controls the false-promotion rate at ``alpha`` under arbitrary stopping.

Deliberately conservative: fixed ``lam`` (not GROW-tuned) and clipped
bets keep every factor strictly positive, so no finite sample can
zero the martingale outright — a bad challenger shrinks toward 0
gradually instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_SCALE_FLOOR = 1e-12
_DEFAULT_INIT_SCALE = 1e-3


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
    causal: the scale and bet at origin ``i`` use only ``d_j`` for
    ``j < i``, so appending future observations can never rewrite a
    reported state.
    """

    lam: float = 0.5
    alpha: float = 0.05
    init_scale: float = _DEFAULT_INIT_SCALE
    _states: list[EProcessState] = field(default_factory=list)
    _diffs: list[float] = field(default_factory=list)
    _log_e: float = 0.0
    _promotion_origin: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must lie in (0, 1)")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not np.isfinite(self.init_scale) or self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive and finite")

    def _predictable_scale(self) -> float:
        if not self._diffs:
            return self.init_scale
        s = float(np.median(np.abs(self._diffs)))
        return s if s > _SCALE_FLOOR else _SCALE_FLOOR

    def update(self, challenger_loss: float, incumbent_loss: float) -> EProcessState:
        """Append one origin's losses and return the new state."""
        c = float(challenger_loss)
        b = float(incumbent_loss)
        if not (np.isfinite(c) and np.isfinite(b)):
            raise ValueError("losses must be finite")
        d = c - b
        g = float(np.clip(d / self._predictable_scale(), -1.0, 1.0))
        e = 1.0 - self.lam * g
        # e in (1-lam, 1+lam) — strictly positive by construction.
        self._log_e += float(np.log(e))
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
        "mean_loss_diff": float(np.mean(c - b)),
        "scale_median": float(np.median(np.abs(c - b))),
        "evidence": [
            "ville_inequality",
            "nonnegative_test_martingale",
            "predictable_scale",
            "stopping_time_valid",
        ],
    }
