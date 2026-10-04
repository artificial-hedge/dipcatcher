import pytest

from quant_fund.research import benches_w1324


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alpacaeval_studies_family",
        "bench_arena_hard_studies_family",
        "bench_judge_bench_studies_family",
        "bench_mt_bench_judge_studies_family",
        "bench_prometheus_eval_studies_family",
        "bench_reward_bench_studies_family",
    ],
)
def test_benches_w1324(fam):
    out = getattr(benches_w1324, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
