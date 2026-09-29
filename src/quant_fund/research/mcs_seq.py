"""Anytime-valid model confidence set — sequential elimination of fleet heads.

The batch MCS (Hansen–Lunde–Nason) answers "which heads survive" after a
fixed sample. This lane maintains a survivor set whose coverage — the
probability it still contains a best head — holds *uniformly over time*
under arbitrary stopping, at level ``alpha``.

Construction (e-value elimination; Ville's inequality + union bound):

- For each ordered pair ``(j, h)``, a ``LossEProcess`` tracks evidence
  that ``j`` strictly beats ``h`` (challenger=j, incumbent=h).
- Head ``h`` is eliminated permanently the first origin at which
  ``max_j e_{j -> h} >= (K - 1) / alpha``. Under "h is optimal" every
  ``e_{j -> h}`` is a nonnegative supermartingale, so
  ``P(h ever eliminated | h optimal) <= (K-1) * alpha/(K-1) = alpha``.
  The survivor set therefore contains an optimal head with probability
  ``>= 1 - alpha`` at *every* origin, however long the fleet runs.

Design choices:

- Pairwise processes, not "each head vs the current leader": the leader
  moves between origins, which breaks predictable betting. Pairwise
  e-processes never read the future.
- Elimination is permanent — "ever crossed" is exactly the event the
  union bound controls; a later dip below threshold cannot resurrect.
- The pair process is instantiated with ``alpha/(K-1)`` so its built-in
  promotion flag trips at precisely the elimination threshold.
- Fail closed: non-finite loss, unknown head, or a missing head raises.

Receipt: ``mcs_seq.v1`` — sealed by callers via ``receipt_v2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from quant_fund.research.evalues import LossEProcess

MCS_SEQ_SCHEMA = "mcs_seq.v1"


@dataclass(frozen=True)
class MCSState:
    """Snapshot of the confidence set after one origin."""

    origin: int
    survivors: tuple[str, ...]
    n_eliminated: int
    newly_eliminated: tuple[str, ...]
    champion: str | None


@dataclass
class AnytimeMCS:
    """Sequential MCS over named heads fed per-origin proper losses.

    ``update(losses)`` appends one origin: ``losses`` maps head → loss
    (lower = better) and must contain every registered head.
    """

    heads: tuple[str, ...]
    alpha: float = 0.05
    lam: float = 0.5
    init_scale: float = 1e-3
    _pairs: dict[tuple[str, str], LossEProcess] = field(default_factory=dict, init=False)
    _eliminated: dict[str, int] = field(default_factory=dict, init=False)
    _states: list[MCSState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.heads) < 2:
            raise ValueError("MCS needs at least two heads")
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("head names must be unique")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        pair_alpha = self.alpha / (len(self.heads) - 1)
        self._pairs = {
            (j, h): LossEProcess(lam=self.lam, alpha=pair_alpha, init_scale=self.init_scale)
            for j, h in permutations(self.heads, 2)
        }

    def update(self, losses: dict[str, float]) -> MCSState:
        """Append one origin of losses for all heads; return the new state."""
        if set(losses) != set(self.heads):
            raise ValueError(
                "losses must cover exactly the registered heads "
                f"{sorted(self.heads)}; got {sorted(losses)}"
            )
        for name, v in losses.items():
            if not np.isfinite(float(v)):
                raise ValueError(f"loss for {name!r} is non-finite: {v}")

        # Fix iteration order so every pair sees origins in the same order.
        for j in self.heads:
            for h in self.heads:
                if j == h:
                    continue
                self._pairs[(j, h)].update(losses[j], losses[h])

        newly: list[str] = []
        for h in self.heads:
            if h in self._eliminated:
                continue
            if any(self._pairs[(j, h)].states[-1].promoted for j in self.heads if j != h):
                self._eliminated[h] = len(self._states)
                newly.append(h)

        survivors = tuple(h for h in self.heads if h not in self._eliminated)
        state = MCSState(
            origin=len(self._states),
            survivors=survivors,
            n_eliminated=len(self._eliminated),
            newly_eliminated=tuple(newly),
            champion=survivors[0] if len(survivors) == 1 else None,
        )
        self._states.append(state)
        return state

    @property
    def survivors(self) -> tuple[str, ...]:
        return tuple(h for h in self.heads if h not in self._eliminated)

    @property
    def eliminated(self) -> dict[str, int]:
        return dict(self._eliminated)

    @property
    def states(self) -> list[MCSState]:
        return list(self._states)


def mcs_report(
    loss_streams: dict[str, Any],
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    init_scale: float = 1e-3,
) -> dict[str, Any]:
    """Receipt-shaped sequential-MCS verdict over per-origin loss streams.

    ``loss_streams`` maps head → equal-length per-origin proper losses
    (lower = better). Fails closed on empty/mismatched/non-finite input.
    """
    heads = sorted(loss_streams)
    if len(heads) < 2:
        raise ValueError("need at least two heads")
    arrays: dict[str, np.ndarray] = {}
    n_obs: int | None = None
    for h in heads:
        a = np.asarray(loss_streams[h], dtype=np.float64).ravel()
        if a.size == 0 or not np.isfinite(a).all():
            raise ValueError(f"stream {h!r} empty or non-finite")
        if n_obs is None:
            n_obs = int(a.size)
        elif a.size != n_obs:
            raise ValueError("all heads must observe the same number of losses")
        arrays[h] = a
    assert n_obs is not None

    mcs = AnytimeMCS(tuple(heads), alpha=alpha, lam=lam, init_scale=init_scale)
    for t in range(n_obs):
        mcs.update({h: float(arrays[h][t]) for h in heads})

    final = mcs.states[-1]
    return {
        "kind": "mcs_seq.v1",
        "alpha": alpha,
        "lam": lam,
        "n_heads": len(heads),
        "n_origins": n_obs,
        "survivors": list(final.survivors),
        "eliminated": {h: int(o) for h, o in mcs.eliminated.items()},
        "n_eliminated": len(mcs.eliminated),
        "champion": final.champion,
        "coverage_guarantee": "P(set contains an optimal head at every origin) >= 1 - alpha",
        "evidence": [
            "ville_inequality",
            "pairwise_supermartingales",
            "union_bound_k_minus_1",
            "permanent_elimination",
            "anytime_valid",
        ],
    }
