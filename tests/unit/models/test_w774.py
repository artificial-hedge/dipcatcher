from quant_fund.models.fluctuation_rw import bench_fluctuation_rw
from quant_fund.models.ladder_epoch import bench_ladder_epoch
from quant_fund.models.maxwell_rw import bench_maxwell_rw
from quant_fund.models.sparc_rw import bench_sparc_rw
from quant_fund.models.spitzer_rw import bench_spitzer_rw
from quant_fund.models.wiener_hopf_rw import bench_wiener_hopf_rw


def test_sparc_rw():
    assert bench_sparc_rw()["synthetic_sparc_rw"] == 1.0


def test_spitzer_rw():
    assert bench_spitzer_rw()["synthetic_spitzer_rw"] == 1.0


def test_fluctuation_rw():
    assert bench_fluctuation_rw()["synthetic_fluctuation_rw"] == 1.0


def test_ladder_epoch():
    assert bench_ladder_epoch()["synthetic_ladder_epoch"] == 1.0


def test_wiener_hopf_rw():
    assert bench_wiener_hopf_rw()["synthetic_wiener_hopf_rw"] == 1.0


def test_maxwell_rw():
    assert bench_maxwell_rw()["synthetic_maxwell_rw"] == 1.0
