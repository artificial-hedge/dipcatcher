from quant_fund.models.atomic_topos import bench_atomic_topos
from quant_fund.models.classifying_topos import (
    bench_classifying_topos,
)
from quant_fund.models.essential_morph import (
    bench_essential_morph,
)
from quant_fund.models.giraud_axiom import bench_giraud_axiom
from quant_fund.models.logical_morph import bench_logical_morph
from quant_fund.models.slice_topos import bench_slice_topos


def test_slice_topos():
    assert bench_slice_topos()["synthetic_slice_topos"] == 1.0


def test_logical_morph():
    assert bench_logical_morph()["synthetic_logical_morph"] == 1.0


def test_classifying_topos():
    assert bench_classifying_topos()["synthetic_classifying_topos"] == 1.0


def test_atomic_topos():
    assert bench_atomic_topos()["synthetic_atomic_topos"] == 1.0


def test_essential_morph():
    assert bench_essential_morph()["synthetic_essential_morph"] == 1.0


def test_giraud_axiom():
    assert bench_giraud_axiom()["synthetic_giraud_axiom"] == 1.0
