from quant_fund.models.ad_period import bench_ad_period
from quant_fund.models.b_drb import bench_b_drb
from quant_fund.models.fontaine_curve import bench_fontaine_curve
from quant_fund.models.perfectoid_c import bench_perfectoid_c
from quant_fund.models.phi_mod import bench_phi_mod
from quant_fund.models.untilt import bench_untilt


def test_fontaine_curve():
    assert bench_fontaine_curve()["synthetic_fontaine_curve"] == 1.0


def test_untilt():
    assert bench_untilt()["synthetic_untilt"] == 1.0


def test_perfectoid_c():
    assert bench_perfectoid_c()["synthetic_perfectoid_c"] == 1.0


def test_b_drb():
    assert bench_b_drb()["synthetic_b_drb"] == 1.0


def test_phi_mod():
    assert bench_phi_mod()["synthetic_phi_mod"] == 1.0


def test_ad_period():
    assert bench_ad_period()["synthetic_ad_period"] == 1.0
