"""d'Alembert wave-equation probes."""

from __future__ import annotations

import numpy as np

from quant_fund.models.wave_dalembert import bench_wave_dalembert, dalembert


def test_bench_score():
    assert bench_wave_dalembert()["synthetic_wave_dalembert"] == 1.0


def test_integral_past_default_grid_edge():
    """The g-integral grid used to be hardcoded to [-20, 20] — arguments
    beyond it were silently clipped, corrupting the solution."""
    f = lambda x: np.zeros_like(np.asarray(x, dtype=float))  # noqa: E731
    g = lambda x: np.asarray(x, dtype=float)  # noqa: E731
    # x=19, t=2, c=1 -> integral over [17, 21] of s ds = (441-289)/2 = 76
    u = dalembert(f, g, np.array([19.0]), 2.0, 1.0)
    assert abs(float(u[0]) - 38.0) < 0.1  # old clip gave 27.75
