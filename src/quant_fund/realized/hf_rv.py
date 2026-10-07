"""HF-RV — high-frequency realized variance scaffolding.

Pinned by the Day Wave 140 design drop. See :mod:`docs.HF_RV_DESIGN` for the
full contract (data ingest, PIT gate, jump test, plug-in points, test
surface, open questions).

This module is **scaffolding only**: the public surface is pinned, the
scaffolding function raises :class:`NotImplementedError`, and a follow-up
implementation wave is expected to land the body. The function signature
and dataclass are frozen by the tests in
:mod:`tests.test_hf_rv_scaffolding` — any change to either is a contract
change and must update the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

__all__ = ["HFRVResult", "compute_hf_rv"]


@dataclass
class HFRVResult:
    """Realized variance on a 5-min grid for a single name at a single asof.

    Attributes
    ----------
    rv_5min
        Realized variance on the 5-min grid for the trailing window.
    bpv
        Bipower variation (jump-robust diffusive variance).
    jump_stat
        Max |Lee–Mykland| statistic over the window.
    n_jumps
        Count of 5-min bars flagged as jumps at significance 1%.
    rv_5min_jump_clean
        Realized variance excluding the flagged bars.
    n_obs
        Number of 5-min bars used in the estimate.
    asof
        Decision origin timestamp (PIT).
    available_time
        Latest observable restatement of any input bar.
    source
        Authority-ordered source name: ``"cls"`` / ``"binance"`` / ``"imf"``.
    source_secondary
        Cross-check source if present, else ``None``.
    clock_drift_seconds
        Maximum per-bar clock drift observed in the input, in seconds.
    holes_count
        Number of single-bar holes that were linearly interpolated.
    honest
        ``False`` when the result must NOT be used downstream (fail-closed).
    """

    rv_5min: float
    bpv: float
    jump_stat: float
    n_jumps: int
    rv_5min_jump_clean: float
    n_obs: int
    asof: pd.Timestamp
    available_time: pd.Timestamp
    source: str
    source_secondary: str | None
    clock_drift_seconds: float
    holes_count: int
    honest: bool


#: Default 5-min window: 1 trading day = 288 bars at 5-min granularity.
DEFAULT_WINDOW_BARS: Final[int] = 288

#: Default tick-subsampling stride (5-tick sub-sample).
DEFAULT_SUBSAMPLE_STRIDE: Final[int] = 5

#: Scaffolding error message — referenced by tests.
_NOT_IMPLEMENTED_MSG: Final[str] = (
    "hf_rv_committed: see docs/HF_RV_DESIGN.md. "
    "Scaffolding only; implementation lands in a follow-up wave."
)


def compute_hf_rv(
    bars: pd.DataFrame,
    asof: pd.Timestamp,
    available_time: pd.Timestamp,
    *,
    window_bars: int = DEFAULT_WINDOW_BARS,
    apply_tick_subsample: bool = True,
    subsample_stride: int = DEFAULT_SUBSAMPLE_STRIDE,
    rng_seed: int | None = None,
) -> HFRVResult:
    """Compute HF realized variance on a 5-min grid (scaffolding).

    See :mod:`docs.HF_RV_DESIGN` for the full contract. The implementation
    is intentionally deferred; this scaffold pins the public surface so a
    follow-up wave can land the body without breaking callers.

    Parameters
    ----------
    bars
        5-min bars with columns ``event_time``, ``available_time``, ``open``,
        ``high``, ``low``, ``close`` (and optionally ``volume``). tz-aware
        ``event_time`` in UTC.
    asof
        Decision origin timestamp (PIT).
    available_time
        Latest observable restatement; bars with ``available_time > asof``
        are dropped (PIT gate).
    window_bars
        Number of 5-min bars in the trailing window. Default 288 (1 day).
    apply_tick_subsample
        If ``True``, apply deterministic tick subsampling to reduce
        microstructure noise. Default ``True``.
    subsample_stride
        Stride for tick subsampling. Default 5.
    rng_seed
        Optional seed for tick subsampling; defaults to a deterministic
        seed derived from ``asof``.

    Raises
    ------
    NotImplementedError
        Always. This is scaffolding — see the design doc.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
