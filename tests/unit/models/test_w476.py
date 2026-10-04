from quant_fund.models.adams_novikov import bench_adams_novikov
from quant_fund.models.bp_spectrum import bench_bp_spectrum
from quant_fund.models.greek_letter import bench_greek_letter
from quant_fund.models.landweber_exact import bench_landweber_exact
from quant_fund.models.picard_grp import bench_picard_grp
from quant_fund.models.smith_toda import bench_smith_toda


def test_bp_spectrum():
    assert bench_bp_spectrum()["synthetic_bp_spectrum"] == 1.0


def test_adams_novikov():
    assert bench_adams_novikov()["synthetic_adams_novikov"] == 1.0


def test_landweber_exact():
    assert bench_landweber_exact()["synthetic_landweber_exact"] == 1.0


def test_greek_letter():
    assert bench_greek_letter()["synthetic_greek_letter"] == 1.0


def test_smith_toda():
    assert bench_smith_toda()["synthetic_smith_toda"] == 1.0


def test_picard_grp():
    assert bench_picard_grp()["synthetic_picard_grp"] == 1.0
