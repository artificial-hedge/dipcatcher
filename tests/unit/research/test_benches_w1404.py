import pytest

from quant_fund.research import benches_w1404


@pytest.mark.parametrize(
    "fam",
    [
        "bench_activitynet_qa_studies_family",
        "bench_how2qa_lite_studies_family",
        "bench_movie_qa_lite_studies_family",
        "bench_msrvtt_qa_studies_family",
        "bench_nextqa_lite_studies_family",
        "bench_star_qa_lite_studies_family",
    ],
)
def test_benches_w1404(fam):
    out = getattr(benches_w1404, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
