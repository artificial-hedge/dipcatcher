import pytest

from quant_fund.research import benches_w1365


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arxiv_sum_studies_family",
        "bench_cnn_dailymail_studies_family",
        "bench_dialogsum_lite_studies_family",
        "bench_multi_news_studies_family",
        "bench_pubmed_sum_studies_family",
        "bench_samsum_lite_studies_family",
    ],
)
def test_benches_w1365(fam):
    out = getattr(benches_w1365, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
