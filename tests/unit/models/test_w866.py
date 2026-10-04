from quant_fund.models.deim_point import (
    bench_deim_point,
)
from quant_fund.models.eim_interp import (
    bench_eim_interp,
)
from quant_fund.models.greedy_rb import (
    bench_greedy_rb,
)
from quant_fund.models.pod_galerkin import (
    bench_pod_galerkin,
)
from quant_fund.models.proper_gen import (
    bench_proper_gen,
)
from quant_fund.models.reduced_basis import (
    bench_reduced_basis,
)


def test_pod_galerkin():
    assert bench_pod_galerkin()["synthetic_pod_galerkin"] == 1.0


def test_reduced_basis():
    assert bench_reduced_basis()["synthetic_reduced_basis"] == 1.0


def test_deim_point():
    assert bench_deim_point()["synthetic_deim_point"] == 1.0


def test_greedy_rb():
    assert bench_greedy_rb()["synthetic_greedy_rb"] == 1.0


def test_eim_interp():
    assert bench_eim_interp()["synthetic_eim_interp"] == 1.0


def test_proper_gen():
    assert bench_proper_gen()["synthetic_proper_gen"] == 1.0
