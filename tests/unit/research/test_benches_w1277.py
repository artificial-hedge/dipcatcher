import pytest

from quant_fund.research import benches_w1277


@pytest.mark.parametrize(
    "fam",
    [
        "bench_debate_alignment_studies_family",
        "bench_deliberative_alignment_studies_family",
        "bench_iterated_amplification_studies_family",
        "bench_recursive_reward_studies_family",
        "bench_scalable_oversight_studies_family",
        "bench_weak_to_strong_studies_family",
    ],
)
def test_benches_w1277(fam):
    out = getattr(benches_w1277, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
