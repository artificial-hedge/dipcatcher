import pytest

from quant_fund.research import benches_w1292


@pytest.mark.parametrize(
    "fam",
    [
        "bench_grpo_studies_family",
        "bench_math_reward_studies_family",
        "bench_outcome_reward_studies_family",
        "bench_process_reward_studies_family",
        "bench_rlvr_studies_family",
        "bench_verifiable_reward_studies_family",
    ],
)
def test_benches_w1292(fam):
    out = getattr(benches_w1292, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
