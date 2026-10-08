"""Early-exercise boundary honesty tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models import exercise_boundary as eb
from quant_fund.models.exercise_boundary import bench_exercise_boundary


def test_ref_alignment(monkeypatch):
    """Coarse index i must pair with reference index i*2000/n, not the
    endpoint-inclusive linspace that drifted up to +5 steps."""

    def fake_crr(n: int):
        # boundary as a fraction of step count: bnd[i] = i/n on every tree,
        # so coarse index i pairs exactly with ref index i*2000/n
        return 0.0, np.arange(n, dtype=float) / n

    monkeypatch.setattr(eb, "crr_price", fake_crr)
    out = bench_exercise_boundary()
    assert out["synthetic_boundary_l2"] < 1e-9


def test_bench_passes():
    out = bench_exercise_boundary()
    assert out["synthetic_boundary_l2"] < out["synthetic_boundary_naive_l2"]
