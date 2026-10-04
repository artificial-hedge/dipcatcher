import pytest

from quant_fund.research import benches_w1275


@pytest.mark.parametrize(
    "fam",
    [
        "bench_deliberate_search_studies_family",
        "bench_latent_reasoning_studies_family",
        "bench_self_improvement_studies_family",
        "bench_test_time_scaling_studies_family",
        "bench_tree_thought_studies_family",
        "bench_verifier_gated_studies_family",
    ],
)
def test_benches_w1275(fam):
    out = getattr(benches_w1275, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
