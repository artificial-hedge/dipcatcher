import numpy as np
import pytest

from quant_fund.models.lee_bounds import (
    bench_lee_bounds,
    lee_bounds,
    synth_lee_bounds,
)


def test_bounds_bracket_truth() -> None:
    d = synth_lee_bounds(effect=1.0, sel_shift=0.4, seed=0)
    out = lee_bounds(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["selected"]))
    assert out["bound_lo"] <= 1.0 <= out["bound_hi"]


def test_bounds_widen_with_selection_gap() -> None:
    d_small = synth_lee_bounds(effect=1.0, sel_shift=0.2, seed=1)
    d_big = synth_lee_bounds(effect=1.0, sel_shift=0.6, seed=1)
    o_small = lee_bounds(
        np.asarray(d_small["y"]),
        np.asarray(d_small["treat"]),
        np.asarray(d_small["selected"]),
    )
    o_big = lee_bounds(
        np.asarray(d_big["y"]),
        np.asarray(d_big["treat"]),
        np.asarray(d_big["selected"]),
    )
    assert (o_big["bound_hi"] - o_big["bound_lo"]) > (o_small["bound_hi"] - o_small["bound_lo"])


def test_equal_selection_collapses() -> None:
    d = synth_lee_bounds(sel_shift=0.0, seed=2)
    out = lee_bounds(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["selected"]))
    assert out["bound_hi"] - out["bound_lo"] < 0.3


def test_rates_reported() -> None:
    d = synth_lee_bounds(sel_shift=0.4, seed=3)
    out = lee_bounds(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["selected"]))
    assert out["p1"] > out["p0"]


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 100
    with pytest.raises(ValueError):
        lee_bounds(rng.normal(0, 1, 30), np.zeros(30), np.ones(30))
    with pytest.raises(ValueError):
        lee_bounds(
            rng.normal(0, 1, n) * np.nan,
            rng.binomial(1, 0.5, n).astype(float),
            np.ones(n),
        )
    with pytest.raises(ValueError):
        lee_bounds(
            rng.normal(0, 1, n),
            np.full(n, 2.0),
            np.ones(n),
        )
    with pytest.raises(ValueError):
        lee_bounds(
            rng.normal(0, 1, n),
            rng.binomial(1, 0.5, n).astype(float),
            np.ones(n),  # no selection variation → degenerate
        )


def test_bench() -> None:
    out = bench_lee_bounds()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
