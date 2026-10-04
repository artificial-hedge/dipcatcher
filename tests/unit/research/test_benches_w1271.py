import pytest

from quant_fund.research import benches_w1271


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arena_battle_studies_family",
        "bench_bigbench_studies_family",
        "bench_capability_elicitation_studies_family",
        "bench_contamination_detect_studies_family",
        "bench_helm_eval_studies_family",
        "bench_llm_judge_studies_family",
    ],
)
def test_benches_w1271(fam):
    out = getattr(benches_w1271, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
