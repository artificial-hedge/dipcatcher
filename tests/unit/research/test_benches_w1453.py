import pytest

from quant_fund.research import benches_w1453


@pytest.mark.parametrize(
    "fam",
    [
        "bench_condor_qa_studies_family",
        "bench_harrier_qa_studies_family",
        "bench_kestrel_qa_studies_family",
        "bench_kite_qa_studies_family",
        "bench_osprey_qa_studies_family",
        "bench_vulture_qa_studies_family",
    ],
)
def test_benches_w1453(fam):
    out = getattr(benches_w1453, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
