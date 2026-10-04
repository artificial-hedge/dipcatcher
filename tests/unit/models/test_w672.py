from quant_fund.models.compact_cat import bench_compact_cat
from quant_fund.models.monoidal_derived import (
    bench_monoidal_derived,
)
from quant_fund.models.perverse_cat import bench_perverse_cat
from quant_fund.models.smashing_cat import bench_smashing_cat
from quant_fund.models.super_cat import bench_super_cat
from quant_fund.models.tannakian_cat import bench_tannakian_cat


def test_tannakian_cat():
    assert bench_tannakian_cat()["synthetic_tannakian_cat"] == 1.0


def test_super_cat():
    assert bench_super_cat()["synthetic_super_cat"] == 1.0


def test_perverse_cat():
    assert bench_perverse_cat()["synthetic_perverse_cat"] == 1.0


def test_smashing_cat():
    assert bench_smashing_cat()["synthetic_smashing_cat"] == 1.0


def test_compact_cat():
    assert bench_compact_cat()["synthetic_compact_cat"] == 1.0


def test_monoidal_derived():
    assert bench_monoidal_derived()["synthetic_monoidal_derived"] == 1.0
