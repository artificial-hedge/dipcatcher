import pytest

from quant_fund.research import benches_w1743


@pytest.mark.parametrize(
    "fam",
    [
        "bench_impundulu_qa_studies_family",
        "bench_inkanyamba_qa_studies_family",
        "bench_mamlambo_qa_studies_family",
        "bench_tikoloshe_qa_studies_family",
        "bench_unkulunkulu_qa_studies_family",
        "bench_usilosimapundu_qa_studies_family",
    ],
)
def test_benches_w1743(fam):
    out = getattr(benches_w1743, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
