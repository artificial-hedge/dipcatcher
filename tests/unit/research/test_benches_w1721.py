import pytest

from quant_fund.research import benches_w1721


@pytest.mark.parametrize(
    "fam",
    [
        "bench_albasti_qa_studies_family",
        "bench_erlik_qa_studies_family",
        "bench_shurale_qa_studies_family",
        "bench_suana_qa_studies_family",
        "bench_tengri_qa_studies_family",
        "bench_umai_qa_studies_family",
    ],
)
def test_benches_w1721(fam):
    out = getattr(benches_w1721, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
