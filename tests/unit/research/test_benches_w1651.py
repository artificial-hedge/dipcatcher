import pytest

from quant_fund.research import benches_w1651


@pytest.mark.parametrize(
    "fam",
    [
        "bench_awgy_qa_studies_family",
        "bench_kuritja_qa_studies_family",
        "bench_minka_qa_studies_family",
        "bench_papin_qa_studies_family",
        "bench_yara_qa_studies_family",
        "bench_yowie_qa_studies_family",
    ],
)
def test_benches_w1651(fam):
    out = getattr(benches_w1651, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
