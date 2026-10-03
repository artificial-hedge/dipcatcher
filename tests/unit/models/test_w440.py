from quant_fund.models.bott_period import bench_bott_period
from quant_fund.models.hopf_map import bench_hopf_map
from quant_fund.models.postnikov_twr import bench_postnikov_twr
from quant_fund.models.stable_stem import bench_stable_stem
from quant_fund.models.thom_iso import bench_thom_iso
from quant_fund.models.whitehead_twr import bench_whitehead_twr


def test_thom_iso():
    assert bench_thom_iso()["synthetic_thom_iso"] == 1.0


def test_postnikov_twr():
    assert bench_postnikov_twr()["synthetic_postnikov_twr"] == 1.0


def test_whitehead_twr():
    assert bench_whitehead_twr()["synthetic_whitehead_twr"] == 1.0


def test_bott_period():
    assert bench_bott_period()["synthetic_bott_period"] == 1.0


def test_stable_stem():
    assert bench_stable_stem()["synthetic_stable_stem"] == 1.0


def test_hopf_map():
    assert bench_hopf_map()["synthetic_hopf_map"] == 1.0
