import pytest

from quant_fund.research import benches_w1388


@pytest.mark.parametrize(
    "fam",
    [
        "bench_corpus_qa_studies_family",
        "bench_crag_bench_studies_family",
        "bench_domain_rag_studies_family",
        "bench_freshqa_studies_family",
        "bench_ragas_lite_studies_family",
        "bench_rgb_eval_studies_family",
    ],
)
def test_benches_w1388(fam):
    out = getattr(benches_w1388, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
