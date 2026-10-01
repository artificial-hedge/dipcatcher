import numpy as np
import pytest

from quant_fund.metrics.storey_fdr import (
    bench_storey_fdr,
    fdr_summary,
    storey_pi0,
    storey_qvalues,
    synth_pvals,
)


def test_pi0_estimated() -> None:
    p = synth_pvals(m=400, m1=60, shift=2.5, seed=0)
    assert abs(storey_pi0(p) / 0.85 - 1) < 0.2


def test_pi0_null_near_one() -> None:
    p = synth_pvals(m=400, m1=0, seed=1)
    assert storey_pi0(p) > 0.85


def test_qvalues_recover_alternatives() -> None:
    p = synth_pvals(m=400, m1=60, shift=2.8, seed=2)
    q = storey_qvalues(p)
    assert np.sum(q <= 0.1) >= 35
    assert q[:60].mean() < q[60:].mean()


def test_qvalues_monotone_sorted() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 200)
    q = storey_qvalues(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_fdr_summary_keys() -> None:
    p = synth_pvals(seed=4)
    s = fdr_summary(p)
    assert 0 <= s["pi0"] <= 1
    assert s["n_q"] >= 0


def test_validation() -> None:
    with pytest.raises(ValueError):
        storey_qvalues(np.ones(5))
    with pytest.raises(ValueError):
        storey_pi0(np.array([1.5] * 30))
    with pytest.raises(ValueError):
        storey_qvalues(np.ones(50) * np.nan)
    with pytest.raises(ValueError):
        fdr_summary(np.ones(50) * 0.5, alpha=2.0)


def test_bench() -> None:
    out = bench_storey_fdr()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
