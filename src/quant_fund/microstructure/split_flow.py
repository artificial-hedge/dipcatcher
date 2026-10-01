"""split_flow — order-splitting flow for the ZI-LOB simulator.

The Tóth–Palit–Lillo–Farmer (2015) mechanism for persistent order flow:
institutional **metaorders** are split into many same-sign child market
orders. On the real LOBSTER tape this produces trade-sign lag-1
autocorrelation ~0.7 — an order of magnitude above what a two-state
Markov intensity modulation emits (~0.05), the gap the
``sim_real_ledger`` scorecard measures.

``SplitFlow`` is a drop-in MO driver implementing the same duck-type
interface the simulator consumes (``current() -> RegimeState``,
``advance()``): during a parent it emits ``p_buy`` ∈ {0, 1} (an
all-buy or all-sell regime) for exactly ``K`` consecutive MO events;
between parents it emits the balanced background ``p_buy=0.5`` while a
geometric waiting time runs. Parent size ``K`` is drawn from a clipped
Pareto (heavy-tailed order splitting).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator

from quant_fund.microstructure.zi_lob_simulator import RegimeState

_BACKGROUND = RegimeState("background", 1.0, 0.5)


class SplitFlow:
    """Metaorder-splitting MO driver (Tóth–Lillo–Farmer flow persistence).

    Parameters
    ----------
    p_start : float
        Per-MO-event probability of starting a new parent when idle
        (geometric inter-arrival on the MO clock).
    size_tail : float
        Pareto tail index for parent size ``K`` (~1.5 empirically).
    k_min, k_max : int
        Clip range for parent size in unit-lot children.
    intensity_mult : float
        MO-intensity multiplier while a parent is active (parents are
        usually more aggressive than background).
    purity : float
        Parent-side fill probability while a parent is active. 1.0 (the
        legacy default) emits a pure same-sign stream — every MO inside
        the parent is on the parent's side. On a busy real tape other
        participants' fills interleave between a parent's children, so
        the *fill* run length is shorter than the parent's child count;
        ``purity < 1`` admits that interleaving.
    seed : int
        RNG seed; the flow shares the simulator's determinism contract
        only when given a fixed seed — pass ``rng`` to share a stream.
    """

    def __init__(
        self,
        *,
        p_start: float = 0.02,
        size_tail: float = 1.5,
        k_min: int = 2,
        k_max: int = 200,
        intensity_mult: float = 1.0,
        purity: float = 1.0,
        seed: int = 0,
        rng: Generator | None = None,
    ) -> None:
        if not math.isfinite(p_start) or not 0.0 <= p_start <= 1.0:
            raise ValueError(f"p_start must be a probability, got {p_start!r}")
        if not math.isfinite(size_tail) or size_tail <= 0.0:
            raise ValueError(f"size_tail must be positive, got {size_tail!r}")
        if k_min < 1 or k_max < k_min:
            raise ValueError(f"bad size clip [{k_min}, {k_max}]")
        if not math.isfinite(intensity_mult) or intensity_mult <= 0.0:
            raise ValueError(f"intensity_mult must be > 0, got {intensity_mult!r}")
        if not math.isfinite(purity) or not 0.5 <= purity <= 1.0:
            raise ValueError(f"purity must be in [0.5, 1], got {purity!r}")
        self._p_start = float(p_start)
        self._tail = float(size_tail)
        self._k_min = int(k_min)
        self._k_max = int(k_max)
        self._mult = float(intensity_mult)
        self._purity = float(purity)
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._remaining = 0
        self._side: RegimeState = _BACKGROUND
        self.n_parents = 0
        self.children_sizes: list[int] = []

    def current(self) -> RegimeState:
        return self._side

    def _draw_size(self) -> int:
        # Pareto tail: K = floor(k_min * u^{-1/a}), clipped
        u = float(self._rng.random())
        k = int(math.floor(self._k_min * (1.0 - u) ** (-1.0 / self._tail)))
        return min(self._k_max, max(self._k_min, k))

    def advance(self) -> None:
        """Called once per MO event by the simulator."""
        if self._remaining > 0:
            self._remaining -= 1
            if self._remaining == 0:
                self._side = _BACKGROUND
            return
        if float(self._rng.random()) < self._p_start:
            k = self._draw_size()
            self.children_sizes.append(k)
            self.n_parents += 1
            buy = float(self._rng.random()) < 0.5
            name = "parent_buy" if buy else "parent_sell"
            self._side = RegimeState(name, self._mult, self._purity if buy else 1.0 - self._purity)
            self._remaining = k


def sign_autocorr_curve(signs: np.ndarray, lags: tuple[int, ...]) -> dict[str, float]:
    """Autocorrelation of ±1 signs at multiple lags (n-normalized)."""
    x = np.asarray(signs, dtype=np.float64)
    out: dict[str, float] = {}
    for lag in lags:
        if x.size < lag + 2:
            out[f"lag{lag}"] = float("nan")
            continue
        out[f"lag{lag}"] = float(np.dot(x[:-lag], x[lag:]) / x.size)
    return out
