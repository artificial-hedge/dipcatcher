import pytest

from quant_fund.research import benches_w1823


@pytest.mark.parametrize(
    "fam",
    [
        "bench_almas2_qa_studies_family",
        "bench_khangai2_qa_studies_family",
        "bench_shunu2_qa_studies_family",
        "bench_sulde2_qa_studies_family",
        "bench_tengri2_qa_studies_family",
        "bench_ukerm2_qa_studies_family",
    ],
)
def test_benches_w1823(fam):
    out = getattr(benches_w1823, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
