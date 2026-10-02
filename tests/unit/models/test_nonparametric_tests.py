import numpy as np
import pytest

from quant_fund.models.nonparametric_tests import (
    bench_nonparametric_tests,
    jonckheere_terpstra,
    kruskal_wallis,
    mann_whitney,
    wilcoxon_signed_rank,
)


def test_mwu_rejects_shift():
    rng = np.random.default_rng(0)
    out = mann_whitney(rng.standard_normal(60), rng.standard_normal(60) + 0.8)
    assert out["p_value"] < 0.05
    assert abs(out["cl_es"] - 0.5) > 0.15


def test_mwu_size_on_null():
    rng = np.random.default_rng(1)
    out = mann_whitney(rng.standard_normal(60), rng.standard_normal(60))
    assert out["p_value"] > 0.01


def test_wilcoxon_paired():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(50)
    out = wilcoxon_signed_rank(base + 0.5, base)
    assert out["p_value"] < 0.05


def test_kw_ordered():
    rng = np.random.default_rng(3)
    g = [rng.standard_normal(40) + s for s in (0.0, 0.5, 1.0)]
    out = kruskal_wallis(*g)
    assert out["p_value"] < 0.01
    assert out["k"] == 3.0


def test_jt_ordered():
    rng = np.random.default_rng(4)
    g = [rng.standard_normal(40) + s for s in (0.0, 0.4, 0.8)]
    out = jonckheere_terpstra(*g)
    assert out["p_value"] < 0.05
    assert out["jt_stat"] > 0


def test_jt_null():
    rng = np.random.default_rng(5)
    g = [rng.standard_normal(40) for _ in range(3)]
    out = jonckheere_terpstra(*g)
    assert out["p_value"] > 0.005


def test_input_validation():
    with pytest.raises(ValueError):
        mann_whitney(np.ones(2), np.ones(5))
    with pytest.raises(ValueError):
        wilcoxon_signed_rank(np.ones(10), np.ones(5))
    with pytest.raises(ValueError):
        wilcoxon_signed_rank(np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        kruskal_wallis(np.random.default_rng(0).standard_normal(10))


def test_bench_nonparametric_tests():
    out = bench_nonparametric_tests()
    assert out["score"] == 1.0
    assert out["synthetic_mwu_p_alt"] < 0.01
    assert out["synthetic_jt_p_alt"] < 0.01
    assert out["synthetic_mwu_p_null"] > 0.01
