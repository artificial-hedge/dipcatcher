import pytest

from quant_fund.research import benches_w1845


@pytest.mark.parametrize(
    "fam",
    [
        "bench_allat_qa_studies_family",
        "bench_dushara_qa_studies_family",
        "bench_hubal_qa_studies_family",
        "bench_manat_qa_studies_family",
        "bench_uzza_qa_studies_family",
        "bench_wadd_qa_studies_family",
    ],
)
def test_benches_w1845(fam):
    out = getattr(benches_w1845, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
