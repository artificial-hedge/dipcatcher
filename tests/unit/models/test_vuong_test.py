import numpy as np
import pytest

from quant_fund.models.vuong_test import (
    bench_vuong_test,
    synth_vuong,
    vuong_test,
)


def test_vuong_picks_model1() -> None:
    d = synth_vuong(true_model=1, seed=2)
    out = vuong_test(d["y"], d["x1"], d["x2"])
    assert out["z_vuong"] > 1.96
    assert out["preferred"] == 1.0


def test_vuong_picks_model2() -> None:
    d = synth_vuong(true_model=2, seed=3)
    out = vuong_test(d["y"], d["x1"], d["x2"])
    assert out["z_vuong"] < -1.96
    assert out["preferred"] == -1.0


def test_vuong_distinguishable() -> None:
    d = synth_vuong(true_model=1, seed=4)
    out = vuong_test(d["y"], d["x1"], d["x2"])
    assert out["p_distinguish"] < 0.05


def test_vuong_validation() -> None:
    d = synth_vuong(seed=5)
    with pytest.raises(ValueError):
        vuong_test(d["y"][:50], d["x1"][:50], d["x2"][:50])
    with pytest.raises(ValueError):
        vuong_test(d["y"][:-1], d["x1"], d["x2"])


def test_vuong_deterministic() -> None:
    d = synth_vuong(true_model=1, seed=6)
    a = vuong_test(d["y"], d["x1"], d["x2"])
    b = vuong_test(d["y"], d["x1"], d["x2"])
    assert a["z_vuong"] == b["z_vuong"]


def test_bench_vuong() -> None:
    out = bench_vuong_test()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
