from quant_fund.models.dupont_k import bench_dupont_k
from quant_fund.models.guin_k import bench_guin_k
from quant_fund.models.kodaira_k import bench_kodaira_k
from quant_fund.models.lindenstrauss_k import (
    bench_lindenstrauss_k,
)
from quant_fund.models.suslin_k2 import bench_suslin_k2
from quant_fund.models.tsukada_k import bench_tsukada_k


def test_kodaira_k():
    assert bench_kodaira_k()["synthetic_kodaira_k"] == 1.0


def test_lindenstrauss_k():
    assert bench_lindenstrauss_k()["synthetic_lindenstrauss_k"] == 1.0


def test_tsukada_k():
    assert bench_tsukada_k()["synthetic_tsukada_k"] == 1.0


def test_guin_k():
    assert bench_guin_k()["synthetic_guin_k"] == 1.0


def test_dupont_k():
    assert bench_dupont_k()["synthetic_dupont_k"] == 1.0


def test_suslin_k2():
    assert bench_suslin_k2()["synthetic_suslin_k2"] == 1.0
