from quant_fund.models.cocartesian import bench_cocartesian
from quant_fund.models.homotopy_cat import (
    bench_homotopy_cat,
)
from quant_fund.models.horn_filler import bench_horn_filler
from quant_fund.models.kan_complex import bench_kan_complex
from quant_fund.models.mapping_space import (
    bench_mapping_space,
)
from quant_fund.models.nerve_cat import bench_nerve_cat


def test_kan_complex():
    assert bench_kan_complex()["synthetic_kan_complex"] == 1.0


def test_horn_filler():
    assert bench_horn_filler()["synthetic_horn_filler"] == 1.0


def test_nerve_cat():
    assert bench_nerve_cat()["synthetic_nerve_cat"] == 1.0


def test_mapping_space():
    assert bench_mapping_space()["synthetic_mapping_space"] == 1.0


def test_homotopy_cat():
    assert bench_homotopy_cat()["synthetic_homotopy_cat"] == 1.0


def test_cocartesian():
    assert bench_cocartesian()["synthetic_cocartesian"] == 1.0
