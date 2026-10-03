import numpy as np
import pytest

from quant_fund.models.bai_perron import (
    bench_bai_perron,
    sequential_breaks,
    synth_bai_perron,
)
from quant_fund.models.favar import (
    bench_favar,
    favar_extract,
    favar_fit,
    synth_favar,
)
from quant_fund.models.panel_coint import (
    bench_panel_coint,
    kao_test,
    pedroni_panel_adf,
    synth_panel_coint,
)


def test_bai_perron_two_breaks() -> None:
    d = synth_bai_perron(breaks=(133, 266), seed=2)
    x1 = np.column_stack([np.ones(400), d["x"]])
    out = sequential_breaks(d["y"], x1, m_max=5)
    assert out["breaks"].size == 2
    assert np.abs(np.sort(out["breaks"])[:2] - np.array([133, 266])).max() < 25


def test_bai_perron_no_break() -> None:
    d = synth_bai_perron(breaks=(400,), betas=(0.5,), seed=3)
    x1 = np.column_stack([np.ones(400), d["x"]])
    out = sequential_breaks(d["y"], x1, m_max=5)
    assert out["breaks"].size <= 1


def test_bai_perron_bench() -> None:
    out = bench_bai_perron()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0


def test_favar_factors_explain() -> None:
    d = synth_favar(r=2, seed=4)
    ex = favar_extract(d["x"], 2)
    assert ex["explained"][0] > 0.5


def test_favar_fit_shape() -> None:
    d = synth_favar(r=2, seed=5)
    out = favar_fit(d["y"], d["x"], r=2, p=1)
    assert out["factors"].shape[1] == 2
    assert out["n_var"][0] == 3  # y + 2 factors


def test_favar_validation() -> None:
    with pytest.raises(ValueError):
        favar_extract(np.zeros((20, 3)), 2)


def test_favar_bench() -> None:
    out = bench_favar()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0


def test_panel_coint_rejects_on_cointegrated() -> None:
    d = synth_panel_coint(cointegrated=True, seed=6)
    assert kao_test(d["y"], d["x"])["pvalue"] < 0.05
    assert pedroni_panel_adf(d["y"], d["x"])["pvalue"] < 0.1


def test_panel_coint_bench() -> None:
    out = bench_panel_coint()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
