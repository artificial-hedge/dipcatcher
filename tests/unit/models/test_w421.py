from quant_fund.models.endo_coend import bench_endo_coend
from quant_fund.models.frobenius_alg import bench_frobenius_alg
from quant_fund.models.profunctor_toy import bench_profunctor_toy
from quant_fund.models.span_compose import bench_span_compose
from quant_fund.models.star_autonomous import bench_star_autonomous
from quant_fund.models.traced_monoidal import bench_traced_monoidal


def test_traced_monoidal():
    assert bench_traced_monoidal()["synthetic_traced_monoidal"] == 1.0


def test_star_autonomous():
    assert bench_star_autonomous()["synthetic_star_autonomous"] == 1.0


def test_frobenius_alg():
    assert bench_frobenius_alg()["synthetic_frobenius_alg"] == 1.0


def test_span_compose():
    assert bench_span_compose()["synthetic_span_compose"] == 1.0


def test_profunctor_toy():
    assert bench_profunctor_toy()["synthetic_profunctor_toy"] == 1.0


def test_endo_coend():
    assert bench_endo_coend()["synthetic_endo_coend"] == 1.0
