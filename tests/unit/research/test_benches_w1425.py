import pytest

from quant_fund.research import benches_w1425


@pytest.mark.parametrize(
    "fam",
    [
        "bench_article_qa_studies_family",
        "bench_broadcast_qa_studies_family",
        "bench_column_qa_studies_family",
        "bench_debate_qa_studies_family",
        "bench_editorial_qa_studies_family",
        "bench_headline_qa_studies_family",
    ],
)
def test_benches_w1425(fam):
    out = getattr(benches_w1425, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
