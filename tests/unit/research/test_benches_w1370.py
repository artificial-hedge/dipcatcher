import pytest

from quant_fund.research import benches_w1370


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bbh_lite_studies_family",
        "bench_gpqa_lite_studies_family",
        "bench_if_eval_studies_family",
        "bench_live_bench_studies_family",
        "bench_olympic_bench_studies_family",
        "bench_trivia_qa_lite_studies_family",
    ],
)
def test_benches_w1370(fam):
    out = getattr(benches_w1370, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
