from quant_fund.models.bo_htpy import bench_bo_htpy
from quant_fund.models.chromatic_square import bench_chromatic_square
from quant_fund.models.devinatz_htpy import bench_devinatz_htpy
from quant_fund.models.hopkins_smith import bench_hopkins_smith
from quant_fund.models.morava_stab import bench_morava_stab
from quant_fund.models.telescope_tower import bench_telescope_tower


def test_devinatz_htpy():
    assert bench_devinatz_htpy()["synthetic_devinatz_htpy"] == 1.0


def test_hopkins_smith():
    assert bench_hopkins_smith()["synthetic_hopkins_smith"] == 1.0


def test_morava_stab():
    assert bench_morava_stab()["synthetic_morava_stab"] == 1.0


def test_chromatic_square():
    assert bench_chromatic_square()["synthetic_chromatic_square"] == 1.0


def test_telescope_tower():
    assert bench_telescope_tower()["synthetic_telescope_tower"] == 1.0


def test_bo_htpy():
    assert bench_bo_htpy()["synthetic_bo_htpy"] == 1.0
