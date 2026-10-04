import pytest

from quant_fund.research import benches_w1289


@pytest.mark.parametrize(
    "fam",
    [
        "bench_analogical_prompt_studies_family",
        "bench_cot_studies_family",
        "bench_reflexion_studies_family",
        "bench_scratchpad_studies_family",
        "bench_self_consistency_studies_family",
        "bench_stepwise_verify_studies_family",
    ],
)
def test_benches_w1289(fam):
    out = getattr(benches_w1289, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
