import numpy as np
import pytest

from quant_fund.models.oaxaca_blinder import (
    bench_oaxaca_blinder,
    ob_decompose,
    synth_ob,
)


def test_parts_sum_to_gap() -> None:
    d = synth_ob(seed=0)
    out = ob_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert abs((out["composition"] + out["structure"]) - out["gap"]) < 1e-9


def test_threefold_sums() -> None:
    d = synth_ob(seed=1)
    out = ob_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert abs((out["endow3"] + out["coef3"] + out["interact3"]) - out["gap"]) < 1e-9


def test_zero_gap_case() -> None:
    d = synth_ob(gap_composition=0.0, gap_structure=0.0, seed=2)
    out = ob_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert abs(out["gap"]) < 0.2


def test_composition_positive_when_x_differs() -> None:
    d = synth_ob(gap_composition=0.8, gap_structure=0.0, seed=3)
    out = ob_decompose(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["group"]))
    assert out["composition"] > 0.3


def test_synth_shapes() -> None:
    d = synth_ob(n=500, seed=4)
    assert np.asarray(d["y"]).shape == (500,)
    assert np.asarray(d["x"]).shape == (500, 2)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 100
    with pytest.raises(ValueError):
        ob_decompose(rng.normal(0, 1, 30), rng.normal(0, 1, (30, 2)), np.zeros(30))
    with pytest.raises(ValueError):
        ob_decompose(
            rng.normal(0, 1, n),
            np.column_stack([np.ones(n), rng.normal(0, 1, n)]),
            np.full(n, 2.0),
        )
    with pytest.raises(ValueError):
        ob_decompose(
            rng.normal(0, 1, n) * np.nan,
            np.column_stack([np.ones(n), rng.normal(0, 1, n)]),
            rng.binomial(1, 0.5, n).astype(float),
        )
    with pytest.raises(ValueError):
        ob_decompose(
            rng.normal(0, 1, n),
            np.column_stack([np.ones(n), rng.normal(0, 1, n)]),
            np.array([1.0] * 95 + [0.0] * 5),  # tiny group
        )


def test_bench() -> None:
    out = bench_oaxaca_blinder()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
