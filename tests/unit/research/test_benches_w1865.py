import pytest

from quant_fund.research import benches_w1865


@pytest.mark.parametrize(
    "fam",
    [
        "bench_enid_qa_studies_family",
        "bench_geraint_qa_studies_family",
        "bench_gwalchmei_qa_studies_family",
        "bench_olwen_qa_studies_family",
        "bench_owen_qa_studies_family",
        "bench_rheged_qa_studies_family",
    ],
)
def test_benches_w1865(fam):
    out = getattr(benches_w1865, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
