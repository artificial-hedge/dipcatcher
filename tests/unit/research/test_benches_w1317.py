import pytest

from quant_fund.research import benches_w1317


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arc_eval_studies_family",
        "bench_do_anything_studies_family",
        "bench_step_eval_studies_family",
        "bench_strong_reject_studies_family",
        "bench_verifier_reward_studies_family",
        "bench_winogrande_studies_family",
    ],
)
def test_benches_w1317(fam):
    out = getattr(benches_w1317, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
