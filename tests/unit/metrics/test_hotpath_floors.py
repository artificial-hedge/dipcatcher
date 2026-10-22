"""Durable correctness + speed floors for scoring/bootstrap hot paths.

SYNTHETIC timings only — not market evidence. Mirrors the native-kernel
speedup-floor pattern so a silent revert to the interpreted loops fails CI.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest

from quant_fund.metrics._bootstrap_kernels import HAVE_NUMBA
from quant_fund.metrics.inference import stationary_bootstrap_indices
from quant_fund.metrics.scoring import crps_from_quantiles, pinball_loss

pytestmark = pytest.mark.synthetic


def _median_s(fn, *, repeats: int = 5) -> float:
    fn()
    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()
    return float(np.median(samples))


def _classic_stationary(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Pre-optimization row loop (per-row ``nonzero`` + ``integers``)."""
    p = 1.0 / float(mean_block)
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    positions = np.arange(n)
    out = np.empty((n_boot, n), dtype=np.intp)
    for b in range(n_boot):
        starts = np.nonzero(new_block[b])[0]
        block_id = np.cumsum(new_block[b]) - 1
        origins = rng.integers(0, n, size=starts.size)
        out[b] = (origins[block_id] + positions - starts[block_id]) % n
    return out


def test_stationary_bootstrap_bit_identical_to_classic_row_loop() -> None:
    seed = 20_260_927
    got = stationary_bootstrap_indices(400, 120, 8.0, np.random.default_rng(seed))
    expected = _classic_stationary(400, 120, 8.0, np.random.default_rng(seed))
    assert np.array_equal(got, expected)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba required for bootstrap floor")
def test_stationary_bootstrap_faster_than_classic_row_loop() -> None:
    n, n_boot, mean_block = 3000, 1000, 12.0

    def _fast() -> np.ndarray:
        return stationary_bootstrap_indices(n, n_boot, mean_block, np.random.default_rng(3))

    def _slow() -> np.ndarray:
        return _classic_stationary(n, n_boot, mean_block, np.random.default_rng(3))

    speedup = _median_s(_slow) / _median_s(_fast)
    assert speedup >= 1.3, f"bootstrap speedup {speedup:.2f}x below 1.3x floor"


def test_crps_from_quantiles_fused_matches_pinball_sum() -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(size=2_000)
    taus = np.linspace(0.05, 0.95, 19)
    q = y[:, None] + rng.normal(0.0, 0.4, size=(y.size, taus.size))
    got = crps_from_quantiles(y, q, taus)
    dt = np.diff(np.concatenate([[0.0], taus]))
    total = np.zeros(y.shape[0], dtype=float)
    for k, tau in enumerate(taus):
        total += 2.0 * pinball_loss(y, q[:, k], float(tau)) * dt[k]
    assert got == pytest.approx(float(np.mean(total)), rel=0.0, abs=1e-12)
