from quant_fund.models.foliation import bench_foliation
from quant_fund.models.godbillon_vey import bench_godbillon_vey
from quant_fund.models.haefliger_struct import bench_haefliger_struct
from quant_fund.models.holonomy_grp import bench_holonomy_grp
from quant_fund.models.novikov_thm import bench_novikov_thm
from quant_fund.models.thurston_fol import bench_thurston_fol


def test_foliation():
    assert bench_foliation()["synthetic_foliation"] == 1.0


def test_holonomy_grp():
    assert bench_holonomy_grp()["synthetic_holonomy_grp"] == 1.0


def test_godbillon_vey():
    assert bench_godbillon_vey()["synthetic_godbillon_vey"] == 1.0


def test_haefliger_struct():
    assert bench_haefliger_struct()["synthetic_haefliger_struct"] == 1.0


def test_novikov_thm():
    assert bench_novikov_thm()["synthetic_novikov_thm"] == 1.0


def test_thurston_fol():
    assert bench_thurston_fol()["synthetic_thurston_fol"] == 1.0
