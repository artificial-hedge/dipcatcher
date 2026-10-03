from quant_fund.models.hyperbolic_3mfd import bench_hyperbolic_3mfd
from quant_fund.models.jorgensen_thurston import bench_jorgensen_thurston
from quant_fund.models.kleinian_group import bench_kleinian_group
from quant_fund.models.limit_set import bench_limit_set
from quant_fund.models.mostow_rigidity import bench_mostow_rigidity
from quant_fund.models.tameness_thm import bench_tameness_thm


def test_kleinian_group():
    assert bench_kleinian_group()["synthetic_kleinian_group"] == 1.0


def test_limit_set():
    assert bench_limit_set()["synthetic_limit_set"] == 1.0


def test_hyperbolic_3mfd():
    assert bench_hyperbolic_3mfd()["synthetic_hyperbolic_3mfd"] == 1.0


def test_mostow_rigidity():
    assert bench_mostow_rigidity()["synthetic_mostow_rigidity"] == 1.0


def test_jorgensen_thurston():
    assert bench_jorgensen_thurston()["synthetic_jorgensen_thurston"] == 1.0


def test_tameness_thm():
    assert bench_tameness_thm()["synthetic_tameness_thm"] == 1.0
