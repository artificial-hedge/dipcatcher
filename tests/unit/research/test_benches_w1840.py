import pytest

from quant_fund.research import benches_w1840


@pytest.mark.parametrize(
    "fam",
    [
        "bench_atargatis2_qa_studies_family",
        "bench_chemosh2_qa_studies_family",
        "bench_gad2_qa_studies_family",
        "bench_haddad2_qa_studies_family",
        "bench_mot2_qa_studies_family",
        "bench_qos2_qa_studies_family",
    ],
)
def test_benches_w1840(fam):
    out = getattr(benches_w1840, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
