from quant_fund.models.dehn_surgery import bench_dehn_surgery
from quant_fund.models.heegaard_splitting import bench_heegaard_splitting
from quant_fund.models.normal_surface import bench_normal_surface
from quant_fund.models.sutured_mfd import bench_sutured_mfd
from quant_fund.models.taut_foliation import bench_taut_foliation
from quant_fund.models.thin_position import bench_thin_position


def test_heegaard_splitting():
    assert bench_heegaard_splitting()["synthetic_heegaard_splitting"] == 1.0


def test_dehn_surgery():
    assert bench_dehn_surgery()["synthetic_dehn_surgery"] == 1.0


def test_sutured_mfd():
    assert bench_sutured_mfd()["synthetic_sutured_mfd"] == 1.0


def test_taut_foliation():
    assert bench_taut_foliation()["synthetic_taut_foliation"] == 1.0


def test_thin_position():
    assert bench_thin_position()["synthetic_thin_position"] == 1.0


def test_normal_surface():
    assert bench_normal_surface()["synthetic_normal_surface"] == 1.0
