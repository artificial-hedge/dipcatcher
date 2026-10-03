from quant_fund.models.anosov import bench_anosov
from quant_fund.models.bowen_spec import bench_bowen_spec
from quant_fund.models.horseshoe import bench_horseshoe
from quant_fund.models.markov_partition import bench_markov_partition
from quant_fund.models.srb_measure import bench_srb_measure
from quant_fund.models.stable_mfld import bench_stable_mfld


def test_anosov():
    assert bench_anosov()["synthetic_anosov"] == 1.0


def test_srb_measure():
    assert bench_srb_measure()["synthetic_srb_measure"] == 1.0


def test_horseshoe():
    assert bench_horseshoe()["synthetic_horseshoe"] == 1.0


def test_stable_mfld():
    assert bench_stable_mfld()["synthetic_stable_mfld"] == 1.0


def test_bowen_spec():
    assert bench_bowen_spec()["synthetic_bowen_spec"] == 1.0


def test_markov_partition():
    assert bench_markov_partition()["synthetic_markov_partition"] == 1.0
