import pytest

from quant_fund.research import benches_w1768


@pytest.mark.parametrize(
    "fam",
    [
        "bench_buluku_qa_studies_family",
        "bench_chukwu_qa_studies_family",
        "bench_eshu_qa_studies_family",
        "bench_mawu_qa_studies_family",
        "bench_nyambi_qa_studies_family",
        "bench_oshumare_qa_studies_family",
    ],
)
def test_benches_w1768(fam):
    out = getattr(benches_w1768, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
