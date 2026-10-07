"""Adversarial probes for specification_curve."""

import pytest

from quant_fund.models import specification_curve as sc


def _data(n: int = 300, seed: int = 0):
    d = sc.synth_multiverse(n=n, seed=seed)
    return d["y"], d["treat"], d["covariate"]


def test_all_skipped_grid_raises():
    """Grid whose every spec needs a covariate, but none provided."""
    y, t, _ = _data()
    with pytest.raises(ValueError, match="no estimable"):
        sc.specification_curve(y, t, None, grid=[("top_half", "linear", "level")])


def test_shuffle_rejects_zero_shuffles():
    y, t, c = _data()
    with pytest.raises(ValueError, match="n_shuffles"):
        sc.spec_curve_shuffle_p(y, t, c, n_shuffles=0)


def test_shuffle_fails_when_every_null_fails():
    """Every placebo curve unestimable -> raise, not a fake p=1.0."""
    y, t, _ = _data()
    grid = [("top_half", "linear", "level")]  # needs covariate: all fail
    with pytest.raises(ValueError):
        sc.spec_curve_shuffle_p(y, t, None, n_shuffles=5, grid=grid)


def test_curve_labels_align_with_effects():
    y, t, c = _data()
    out = sc.specification_curve(y, t, c)
    assert out["n_specs"] == float(len(out["labels"]))
    assert out["n_specs"] == float(out["effects"].size)


def test_bench_smoke():
    out = sc.bench_specification_curve()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_n_specs"] >= 1.0
