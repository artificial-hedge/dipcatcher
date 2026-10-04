from quant_fund.models.adem_relations import bench_adem_relations
from quant_fund.models.bar_resolution import bench_bar_resolution
from quant_fund.models.lambda_algebra import bench_lambda_algebra
from quant_fund.models.serre_cartan import bench_serre_cartan
from quant_fund.models.steenrod_algebra import bench_steenrod_algebra
from quant_fund.models.unstable_modules import bench_unstable_modules


def test_steenrod_algebra():
    assert bench_steenrod_algebra()["synthetic_steenrod_algebra"] == 1.0


def test_adem_relations():
    assert bench_adem_relations()["synthetic_adem_relations"] == 1.0


def test_serre_cartan():
    assert bench_serre_cartan()["synthetic_serre_cartan"] == 1.0


def test_unstable_modules():
    assert bench_unstable_modules()["synthetic_unstable_modules"] == 1.0


def test_lambda_algebra():
    assert bench_lambda_algebra()["synthetic_lambda_algebra"] == 1.0


def test_bar_resolution():
    assert bench_bar_resolution()["synthetic_bar_resolution"] == 1.0
