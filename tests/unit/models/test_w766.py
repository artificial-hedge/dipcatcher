from quant_fund.models.bounded_lip import bench_bounded_lip
from quant_fund.models.bracketing_ent import bench_bracketing_ent
from quant_fund.models.dudley_theorem import bench_dudley_theorem
from quant_fund.models.dvoretzky_thm import bench_dvoretzky_thm
from quant_fund.models.varadarajan_thm import (
    bench_varadarajan_thm,
)
from quant_fund.models.vc_class import bench_vc_class


def test_dudley_theorem():
    assert bench_dudley_theorem()["synthetic_dudley_theorem"] == 1.0


def test_varadarajan_thm():
    assert bench_varadarajan_thm()["synthetic_varadarajan_thm"] == 1.0


def test_dvoretzky_thm():
    assert bench_dvoretzky_thm()["synthetic_dvoretzky_thm"] == 1.0


def test_vc_class():
    assert bench_vc_class()["synthetic_vc_class"] == 1.0


def test_bracketing_ent():
    assert bench_bracketing_ent()["synthetic_bracketing_ent"] == 1.0


def test_bounded_lip():
    assert bench_bounded_lip()["synthetic_bounded_lip"] == 1.0
