from quant_fund.models.beilinson_con import bench_beilinson_con
from quant_fund.models.cellular_motive import bench_cellular_motive
from quant_fund.models.levine_morel import bench_levine_morel
from quant_fund.models.mgl_spec import bench_mgl_spec
from quant_fund.models.motivic_pi0 import bench_motivic_pi0
from quant_fund.models.quadratic_k import bench_quadratic_k


def test_levine_morel():
    assert bench_levine_morel()["synthetic_levine_morel"] == 1.0


def test_quadratic_k():
    assert bench_quadratic_k()["synthetic_quadratic_k"] == 1.0


def test_mgl_spec():
    assert bench_mgl_spec()["synthetic_mgl_spec"] == 1.0


def test_cellular_motive():
    assert bench_cellular_motive()["synthetic_cellular_motive"] == 1.0


def test_motivic_pi0():
    assert bench_motivic_pi0()["synthetic_motivic_pi0"] == 1.0


def test_beilinson_con():
    assert bench_beilinson_con()["synthetic_beilinson_con"] == 1.0
