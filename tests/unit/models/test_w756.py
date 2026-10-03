from quant_fund.models.berestycki_gff import bench_berestycki_gff
from quant_fund.models.duplantier_sheffield import (
    bench_duplantier_sheffield,
)
from quant_fund.models.houchmandzadeh_gff import (
    bench_houchmandzadeh_gff,
)
from quant_fund.models.nick_gff import bench_nick_gff
from quant_fund.models.sheffield_miller import (
    bench_sheffield_miller,
)
from quant_fund.models.wiegmann_zabrodin import (
    bench_wiegmann_zabrodin,
)


def test_berestycki_gff():
    assert bench_berestycki_gff()["synthetic_berestycki_gff"] == 1.0


def test_duplantier_sheffield():
    assert bench_duplantier_sheffield()["synthetic_duplantier_sheffield"] == 1.0


def test_houchmandzadeh_gff():
    assert bench_houchmandzadeh_gff()["synthetic_houchmandzadeh_gff"] == 1.0


def test_nick_gff():
    assert bench_nick_gff()["synthetic_nick_gff"] == 1.0


def test_sheffield_miller():
    assert bench_sheffield_miller()["synthetic_sheffield_miller"] == 1.0


def test_wiegmann_zabrodin():
    assert bench_wiegmann_zabrodin()["synthetic_wiegmann_zabrodin"] == 1.0
