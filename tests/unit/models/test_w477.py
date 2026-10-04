from quant_fund.models.arithmetic_ht import bench_arithmetic_ht
from quant_fund.models.berthelo_crys import bench_berthelo_crys
from quant_fund.models.berthelot_rigid import bench_berthelot_rigid
from quant_fund.models.caro_dm import bench_caro_dm
from quant_fund.models.dagger_dm import bench_dagger_dm
from quant_fund.models.spencer_dm import bench_spencer_dm


def test_dagger_dm():
    assert bench_dagger_dm()["synthetic_dagger_dm"] == 1.0


def test_spencer_dm():
    assert bench_spencer_dm()["synthetic_spencer_dm"] == 1.0


def test_caro_dm():
    assert bench_caro_dm()["synthetic_caro_dm"] == 1.0


def test_berthelot_rigid():
    assert bench_berthelot_rigid()["synthetic_berthelot_rigid"] == 1.0


def test_berthelo_crys():
    assert bench_berthelo_crys()["synthetic_berthelo_crys"] == 1.0


def test_arithmetic_ht():
    assert bench_arithmetic_ht()["synthetic_arithmetic_ht"] == 1.0
