from quant_fund.models.brauer_grp import bench_brauer_grp
from quant_fund.models.chow_group import bench_chow_group
from quant_fund.models.milnor_conj import bench_milnor_conj
from quant_fund.models.motivic_coh import bench_motivic_coh
from quant_fund.models.motivic_stem import bench_motivic_stem
from quant_fund.models.voevodsky_dm import bench_voevodsky_dm


def test_motivic_coh():
    assert bench_motivic_coh()["synthetic_motivic_coh"] == 1.0


def test_chow_group():
    assert bench_chow_group()["synthetic_chow_group"] == 1.0


def test_milnor_conj():
    assert bench_milnor_conj()["synthetic_milnor_conj"] == 1.0


def test_voevodsky_dm():
    assert bench_voevodsky_dm()["synthetic_voevodsky_dm"] == 1.0


def test_motivic_stem():
    assert bench_motivic_stem()["synthetic_motivic_stem"] == 1.0


def test_brauer_grp():
    assert bench_brauer_grp()["synthetic_brauer_grp"] == 1.0
