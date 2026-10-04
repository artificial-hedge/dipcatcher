import pytest

from quant_fund.research import benches_w1288


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adversarial_irl_studies_family",
        "bench_behavior_cloning_studies_family",
        "bench_dagger_studies_family",
        "bench_offline_distill_studies_family",
        "bench_preference_irl_studies_family",
        "bench_skill_extraction_studies_family",
    ],
)
def test_benches_w1288(fam):
    out = getattr(benches_w1288, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
