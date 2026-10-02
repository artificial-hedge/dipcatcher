"""Shared helpers for wave-173 diffusion-exotics canon: reuse the
regime-window fixture + MMD from flow_matching_ts; tiny velocity MLP.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.flow_matching_ts import _mmd, synth_regime_windows

__all__ = ["_mmd", "synth_regime_windows", "gauss_mmd"]


def gauss_mmd(rng: np.random.Generator, Xte: np.ndarray, n: int = 300) -> float:
    return _mmd(rng.standard_normal((n, Xte.shape[1])).astype(np.float64), Xte, bw=1.0)
