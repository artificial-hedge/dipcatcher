from quant_fund.models.conjugate_fil import (
    bench_conjugate_fil,
)
from quant_fund.models.crys_cohom import bench_crys_cohom
from quant_fund.models.divided_power import (
    bench_divided_power,
)
from quant_fund.models.nygaard_filt import (
    bench_nygaard_filt,
)
from quant_fund.models.pd_envelope import bench_pd_envelope
from quant_fund.models.syntomic import bench_syntomic


def test_crys_cohom():
    assert bench_crys_cohom()["synthetic_crys_cohom"] == 1.0


def test_syntomic():
    assert bench_syntomic()["synthetic_syntomic"] == 1.0


def test_divided_power():
    assert bench_divided_power()["synthetic_divided_power"] == 1.0


def test_pd_envelope():
    assert bench_pd_envelope()["synthetic_pd_envelope"] == 1.0


def test_nygaard_filt():
    assert bench_nygaard_filt()["synthetic_nygaard_filt"] == 1.0


def test_conjugate_fil():
    assert bench_conjugate_fil()["synthetic_conjugate_fil"] == 1.0
