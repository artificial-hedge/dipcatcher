import pytest

from quant_fund.research import benches_w1273


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alignment_eval_studies_family",
        "bench_guardrail_studies_family",
        "bench_hallucination_detect_studies_family",
        "bench_jailbreak_defense_studies_family",
        "bench_red_team_studies_family",
        "bench_sleeper_agent_studies_family",
    ],
)
def test_benches_w1273(fam):
    out = getattr(benches_w1273, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
