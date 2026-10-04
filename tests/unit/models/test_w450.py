from quant_fund.models.exact_seq import bench_exact_seq
from quant_fund.models.smash_monoidal import bench_smash_monoidal
from quant_fund.models.spectra_cat import bench_spectra_cat
from quant_fund.models.stable_infty import bench_stable_infty
from quant_fund.models.stable_tstruct import bench_stable_tstruct
from quant_fund.models.thh_tc import bench_thh_tc


def test_stable_infty():
    assert bench_stable_infty()["synthetic_stable_infty"] == 1.0


def test_spectra_cat():
    assert bench_spectra_cat()["synthetic_spectra_cat"] == 1.0


def test_exact_seq():
    assert bench_exact_seq()["synthetic_exact_seq"] == 1.0


def test_stable_tstruct():
    assert bench_stable_tstruct()["synthetic_stable_tstruct"] == 1.0


def test_smash_monoidal():
    assert bench_smash_monoidal()["synthetic_smash_monoidal"] == 1.0


def test_thh_tc():
    assert bench_thh_tc()["synthetic_thh_tc"] == 1.0
