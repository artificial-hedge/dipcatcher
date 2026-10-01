import numpy as np
import pytest

from quant_fund.models.star_model import (
    bench_star,
    linearity_lm3,
    star_fit,
    synth_star,
)


def test_bench_star_passes():
    r = bench_star()
    assert r["score"] == 1.0


def test_lm3_rejects_star():
    y, _ = synth_star(seed=4)
    r = linearity_lm3(y, p=2, d=1)
    assert r["reject"] == 1.0


def test_lm3_accepts_linear():
    _, w = synth_star(seed=6)
    r = linearity_lm3(w, p=2, d=1)
    assert r["reject"] == 0.0


def test_star_recovers_c():
    y, _ = synth_star(seed=8)
    r = star_fit(y, p=1, d=1, kind="lstar")
    assert abs(r["c"]) < 0.4


def test_estar_mode_runs():
    y, _ = synth_star(seed=1)
    r = star_fit(y, p=1, d=1, kind="estar")
    assert np.isfinite(r["gamma"])


def test_rejects_bad_kind():
    y, _ = synth_star(seed=1)
    with pytest.raises(ValueError):
        star_fit(y, kind="bad")
