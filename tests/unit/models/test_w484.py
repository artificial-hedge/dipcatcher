from quant_fund.models.formal_model import bench_formal_model
from quant_fund.models.internal_univ import bench_internal_univ
from quant_fund.models.stein_space import bench_stein_space
from quant_fund.models.synth_stable import bench_synth_stable
from quant_fund.models.univalent_found import bench_univalent_found
from quant_fund.models.virtual_hodge import bench_virtual_hodge


def test_internal_univ():
    assert bench_internal_univ()["synthetic_internal_univ"] == 1.0


def test_virtual_hodge():
    assert bench_virtual_hodge()["synthetic_virtual_hodge"] == 1.0


def test_stein_space():
    assert bench_stein_space()["synthetic_stein_space"] == 1.0


def test_formal_model():
    assert bench_formal_model()["synthetic_formal_model"] == 1.0


def test_univalent_found():
    assert bench_univalent_found()["synthetic_univalent_found"] == 1.0


def test_synth_stable():
    assert bench_synth_stable()["synthetic_synth_stable"] == 1.0
