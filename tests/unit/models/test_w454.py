from quant_fund.models.decomp_thm import bench_decomp_thm
from quant_fund.models.fourier_sato import bench_fourier_sato
from quant_fund.models.ic_stalk import bench_ic_stalk
from quant_fund.models.middle_ext import bench_middle_ext
from quant_fund.models.riemann_hilbert import bench_riemann_hilbert
from quant_fund.models.vanishing_cycles import bench_vanishing_cycles


def test_ic_stalk():
    assert bench_ic_stalk()["synthetic_ic_stalk"] == 1.0


def test_decomp_thm():
    assert bench_decomp_thm()["synthetic_decomp_thm"] == 1.0


def test_riemann_hilbert():
    assert bench_riemann_hilbert()["synthetic_riemann_hilbert"] == 1.0


def test_fourier_sato():
    assert bench_fourier_sato()["synthetic_fourier_sato"] == 1.0


def test_vanishing_cycles():
    assert bench_vanishing_cycles()["synthetic_vanishing_cycles"] == 1.0


def test_middle_ext():
    assert bench_middle_ext()["synthetic_middle_ext"] == 1.0
