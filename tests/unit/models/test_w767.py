from quant_fund.models.borell_tis import bench_borell_tis
from quant_fund.models.fernique_thm import bench_fernique_thm
from quant_fund.models.gordon_thm import bench_gordon_thm
from quant_fund.models.slepian_lemma import bench_slepian_lemma
from quant_fund.models.sudakov_min import bench_sudakov_min
from quant_fund.models.talagrand_conc import (
    bench_talagrand_conc,
)


def test_slepian_lemma():
    assert bench_slepian_lemma()["synthetic_slepian_lemma"] == 1.0


def test_fernique_thm():
    assert bench_fernique_thm()["synthetic_fernique_thm"] == 1.0


def test_borell_tis():
    assert bench_borell_tis()["synthetic_borell_tis"] == 1.0


def test_sudakov_min():
    assert bench_sudakov_min()["synthetic_sudakov_min"] == 1.0


def test_talagrand_conc():
    assert bench_talagrand_conc()["synthetic_talagrand_conc"] == 1.0


def test_gordon_thm():
    assert bench_gordon_thm()["synthetic_gordon_thm"] == 1.0
