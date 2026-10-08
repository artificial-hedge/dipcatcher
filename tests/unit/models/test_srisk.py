import pytest

from quant_fund.models.srisk import bench_srisk, lrmes, mes, srisk, synth_srisk


def test_bench_srisk_passes():
    r = bench_srisk()
    assert r["synthetic_score"] == 1.0


def test_mes_high_beta_lower():
    hi, lo, mkt = synth_srisk(seed=4)
    assert mes(hi, mkt)["mes"] < mes(lo, mkt)["mes"]


def test_mes_counts_tail_days():
    hi, _, mkt = synth_srisk(seed=2)
    r = mes(hi, mkt, alpha=0.05)
    assert r["n_tail_days"] >= 3


def test_srisk_floors_at_zero():
    _, lo, mkt = synth_srisk(seed=1)
    r = srisk(lo, mkt, debt=10.0, equity=100.0)
    assert r["srisk"] == 0.0


def test_lrmes_more_negative_for_high_beta():
    hi, lo, mkt = synth_srisk(seed=3)
    assert lrmes(hi, mkt) > lrmes(lo, mkt)


def test_srisk_rejects_bad_sheet():
    hi, _, mkt = synth_srisk(seed=1)
    with pytest.raises(ValueError):
        srisk(hi, mkt, debt=-1.0, equity=10.0)
