from quant_fund.models.baker_thm import bench_baker_thm
from quant_fund.models.gelfond_schneider import bench_gelfond_schneider
from quant_fund.models.hermite_lindemann import bench_hermite_lindemann
from quant_fund.models.lindemann_weier import bench_lindemann_weier
from quant_fund.models.schanuel_conj import bench_schanuel_conj
from quant_fund.models.siegel_shidlovskii import bench_siegel_shidlovskii


def test_hermite_lindemann():
    assert bench_hermite_lindemann()["synthetic_hermite_lindemann"] == 1.0


def test_gelfond_schneider():
    assert bench_gelfond_schneider()["synthetic_gelfond_schneider"] == 1.0


def test_baker_thm():
    assert bench_baker_thm()["synthetic_baker_thm"] == 1.0


def test_lindemann_weier():
    assert bench_lindemann_weier()["synthetic_lindemann_weier"] == 1.0


def test_schanuel_conj():
    assert bench_schanuel_conj()["synthetic_schanuel_conj"] == 1.0


def test_siegel_shidlovskii():
    assert bench_siegel_shidlovskii()["synthetic_siegel_shidlovskii"] == 1.0
