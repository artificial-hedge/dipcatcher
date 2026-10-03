import numpy as np
import pytest

from quant_fund.models.binscatter import (
    bench_binscatter,
    binscatter,
    binspec_test,
    synth_cef,
)


def test_bins_cover_and_equal_mass() -> None:
    d = synth_cef(seed=0)
    out = binscatter(np.asarray(d["x"]), np.asarray(d["y"]), j=10)
    cnt = np.asarray(out["count"])
    assert cnt.sum() == 3000
    assert cnt.min() > 250  # roughly equal-mass


def test_kink_rejects_linear() -> None:
    d = synth_cef(kind="kink", seed=1)
    out = binspec_test(np.asarray(d["x"]), np.asarray(d["y"]), degree=1)
    assert out["p"] < 0.01


def test_linear_accepted() -> None:
    d = synth_cef(kind="linear", seed=2)
    out = binspec_test(np.asarray(d["x"]), np.asarray(d["y"]), degree=1)
    assert out["p"] > 0.05


def test_bin_means_track_cef() -> None:
    d = synth_cef(kind="kink", seed=3)
    out = binscatter(np.asarray(d["x"]), np.asarray(d["y"]), j=10)
    # leftmost bin mean well below rightmost (kinked up)
    yb = np.asarray(out["y_bar"])
    assert yb[-1] > yb[0]


def test_synth_shapes() -> None:
    d = synth_cef(n=200, seed=4)
    assert np.asarray(d["x"]).shape == (200,)
    with pytest.raises(ValueError):
        synth_cef(n=50, kind="nope")


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        binscatter(rng.normal(0, 1, 30), rng.normal(0, 1, 30))
    with pytest.raises(ValueError):
        binscatter(rng.normal(0, 1, 100), rng.normal(0, 1, 100), j=40)
    with pytest.raises(ValueError):
        binscatter(rng.normal(0, 1, 100) * np.nan, rng.normal(0, 1, 100))
    with pytest.raises(ValueError):
        binspec_test(rng.normal(0, 1, 100), rng.normal(0, 1, 100), degree=9)


def test_bench() -> None:
    out = bench_binscatter()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
