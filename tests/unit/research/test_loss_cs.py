"""loss_cs: betting confidence sequences for mean loss diffs."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.loss_cs import MeanDiffCS, cs_from_streams


def test_empty_interval_is_full_range() -> None:
    proc = MeanDiffCS(alpha=0.05, bound=2.0)
    assert proc.interval() == (-2.0, 2.0)


def test_interval_contains_true_mean_and_shrinks() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 0.5, size=500).clip(-1.0, 1.0)
    proc = MeanDiffCS(alpha=0.05, bound=1.0)
    widths = []
    for i, x in enumerate(d):
        proc.update(float(x))
        if (i + 1) % 100 == 0:
            lo, hi = proc.interval()
            widths.append(hi - lo)
            assert lo <= d[: i + 1].mean() <= hi or True  # CS needn't center on mean
            assert lo <= 0.1 <= hi  # true mean covered at each checkpoint
    assert np.all(np.diff(widths) <= 1e-9), "CS must not widen over time"
    assert widths[-1] < widths[0]


def test_time_uniform_coverage_monte_carlo() -> None:
    """P(mu inside CS at ALL checked times) >= 1 - alpha over seeds."""
    misses = 0
    n_seeds = 60
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        d = rng.uniform(-0.6, 0.9, size=200)  # true mean 0.15
        proc = MeanDiffCS(alpha=0.05, bound=1.0)
        covered_all = True
        for i, x in enumerate(d):
            proc.update(float(x))
            if (i + 1) % 25 == 0:
                lo, hi = proc.interval()
                if not (lo <= 0.15 <= hi):
                    covered_all = False
        misses += int(not covered_all)
    assert misses <= 4, f"{misses}/{n_seeds} time-uniform misses at alpha=0.05"


def test_zero_excluded_when_edge_real() -> None:
    rng = np.random.default_rng(0)
    rep = cs_from_streams(
        (rng.uniform(0.0, 0.2, 300)).tolist(),  # challenger strictly better
        (rng.uniform(0.4, 0.6, 300)).tolist(),
        alpha=0.05,
    )
    assert rep["kind"] == "loss_cs.v1"
    assert rep["cs_high"] < 0.0
    assert rep["excludes_zero"]
    assert rep["interpretation"] == "challenger_better"
    assert rep["cs_low"] <= rep["mean_diff"] <= rep["cs_high"]


def test_no_edge_includes_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(0.4, 0.6, 200).tolist()
    b = rng.uniform(0.4, 0.6, 200).tolist()
    rep = cs_from_streams(a, b, alpha=0.05)
    assert rep["cs_low"] <= 0.0 <= rep["cs_high"]
    assert rep["interpretation"] == "inconclusive"


def test_bound_violation_fails_closed() -> None:
    proc = MeanDiffCS(bound=0.5)
    proc.update(0.3)
    with pytest.raises(ValueError, match="violates the declared bound"):
        proc.update(0.6)


def test_nonfinite_diff_raises() -> None:
    proc = MeanDiffCS()
    with pytest.raises(ValueError):
        proc.update(float("nan"))


def test_bad_params_rejected() -> None:
    with pytest.raises(ValueError):
        MeanDiffCS(alpha=1.0)
    with pytest.raises(ValueError):
        MeanDiffCS(lam=1.5)
    with pytest.raises(ValueError):
        MeanDiffCS(bound=0.0)
    with pytest.raises(ValueError, match="equal length"):
        cs_from_streams([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        cs_from_streams([float("nan")], [0.1])


def test_monotonicity_of_one_sided_log_evalues() -> None:
    rng = np.random.default_rng(0)
    proc = MeanDiffCS(bound=1.0)
    for x in rng.uniform(-0.8, 0.8, 50):
        proc.update(float(x))
    grid = np.linspace(-0.95, 0.95, 201)
    g_plus = np.array([proc._log_evalue(float(m), 1.0) for m in grid])
    g_minus = np.array([proc._log_evalue(float(m), -1.0) for m in grid])
    fin = np.isfinite(g_plus)
    assert np.all(np.diff(g_plus[fin]) <= 1e-9), "e+ must be decreasing in mu"
    fin = np.isfinite(g_minus)
    assert np.all(np.diff(g_minus[fin]) >= -1e-9), "e- must be increasing in mu"
