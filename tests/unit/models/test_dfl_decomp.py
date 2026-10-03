import numpy as np
import pytest

from quant_fund.models.dfl_decomp import (
    bench_dfl_decomp,
    dfl_decompose,
    synth_dfl,
)


def test_parts_sum_to_gap() -> None:
    d = synth_dfl(seed=0)
    out = dfl_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert abs(out["structure"] + out["composition"] - out["gap"]) < 1e-9


def test_structure_recovery() -> None:
    d = synth_dfl(struct_shift=0.5, seed=1)
    out = dfl_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert abs(out["structure"] - 0.5) < 0.25


def test_counterfactual_moves_median() -> None:
    d = synth_dfl(comp_shift=0.8, seed=2)
    out = dfl_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert out["q0.50_cf"] > out["q0.50_b"] + 0.5


def test_no_shift_small_gap() -> None:
    d = synth_dfl(comp_shift=0.0, struct_shift=0.0, seed=3)
    out = dfl_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert abs(out["gap"]) < 0.2


def test_synth_shapes() -> None:
    d = synth_dfl(n=200, seed=4)
    assert np.asarray(d["y"]).shape == (200,)
    assert np.asarray(d["x"]).shape == (200, 2)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 100
    x = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    with pytest.raises(ValueError):
        dfl_decompose(rng.normal(0, 1, 30), x[:30], np.zeros(30))
    with pytest.raises(ValueError):
        dfl_decompose(rng.normal(0, 1, n), x, np.full(n, 2.0))
    with pytest.raises(ValueError):
        dfl_decompose(
            rng.normal(0, 1, n) * np.nan,
            x,
            rng.binomial(1, 0.5, n).astype(float),
        )
    with pytest.raises(ValueError):
        dfl_decompose(
            rng.normal(0, 1, n),
            x,
            np.array([1.0] * 96 + [0.0] * 4),
        )


def test_bench() -> None:
    out = bench_dfl_decomp()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
