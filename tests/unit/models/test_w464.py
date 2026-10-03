from quant_fund.models.chromatic_conv import bench_chromatic_conv
from quant_fund.models.k_n_local import bench_k_n_local
from quant_fund.models.morava_e import bench_morava_e
from quant_fund.models.nilpotence_dev import bench_nilpotence_dev
from quant_fund.models.telescopic import bench_telescopic
from quant_fund.models.tmf_spectrum import bench_tmf_spectrum


def test_morava_e():
    assert bench_morava_e()["synthetic_morava_e"] == 1.0


def test_tmf_spectrum():
    assert bench_tmf_spectrum()["synthetic_tmf_spectrum"] == 1.0


def test_k_n_local():
    assert bench_k_n_local()["synthetic_k_n_local"] == 1.0


def test_chromatic_conv():
    assert bench_chromatic_conv()["synthetic_chromatic_conv"] == 1.0


def test_nilpotence_dev():
    assert bench_nilpotence_dev()["synthetic_nilpotence_dev"] == 1.0


def test_telescopic():
    assert bench_telescopic()["synthetic_telescopic"] == 1.0
