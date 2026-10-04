import pytest

from quant_fund.research import benches_w1296


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ensemble_rm_studies_family",
        "bench_judge_reward_studies_family",
        "bench_margin_reward_studies_family",
        "bench_reward_hacking_studies_family",
        "bench_reward_uncertainty_studies_family",
        "bench_rm_btd_studies_family",
    ],
)
def test_benches_w1296(fam):
    out = getattr(benches_w1296, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
