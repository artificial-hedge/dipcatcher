import pytest

from quant_fund.research import benches_w1346


@pytest.mark.parametrize(
    "fam",
    [
        "bench_grail_qa_studies_family",
        "bench_graph_questions_studies_family",
        "bench_kqa_pro_studies_family",
        "bench_lc_quad_studies_family",
        "bench_mintaka_qa_studies_family",
        "bench_spinach_qa_studies_family",
    ],
)
def test_benches_w1346(fam):
    out = getattr(benches_w1346, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
