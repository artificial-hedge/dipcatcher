import pytest

from quant_fund.research import benches_w1802


@pytest.mark.parametrize(
    "fam",
    [
        "bench_awhi2_qa_studies_family",
        "bench_hine3_qa_studies_family",
        "bench_kaikoura_qa_studies_family",
        "bench_moana2_qa_studies_family",
        "bench_ranginui2_qa_studies_family",
        "bench_tanemahuta2_qa_studies_family",
    ],
)
def test_benches_w1802(fam):
    out = getattr(benches_w1802, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
