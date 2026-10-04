from quant_fund.models.anti_self_dual import bench_anti_self_dual
from quant_fund.models.higgs_bundle import bench_higgs_bundle
from quant_fund.models.instanton_moduli import bench_instanton_moduli
from quant_fund.models.kapustin_witten import bench_kapustin_witten
from quant_fund.models.nahm_transform import bench_nahm_transform
from quant_fund.models.yang_mills import bench_yang_mills


def test_yang_mills():
    assert bench_yang_mills()["synthetic_yang_mills"] == 1.0


def test_instanton_moduli():
    assert bench_instanton_moduli()["synthetic_instanton_moduli"] == 1.0


def test_anti_self_dual():
    assert bench_anti_self_dual()["synthetic_anti_self_dual"] == 1.0


def test_higgs_bundle():
    assert bench_higgs_bundle()["synthetic_higgs_bundle"] == 1.0


def test_kapustin_witten():
    assert bench_kapustin_witten()["synthetic_kapustin_witten"] == 1.0


def test_nahm_transform():
    assert bench_nahm_transform()["synthetic_nahm_transform"] == 1.0
