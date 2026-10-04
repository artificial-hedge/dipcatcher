import pytest

from quant_fund.research import benches_w1299


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agent_harm_studies_family",
        "bench_harm_bench_studies_family",
        "bench_jailbreak_bench_studies_family",
        "bench_prompt_inject_studies_family",
        "bench_safety_bench_studies_family",
        "bench_xstest_studies_family",
    ],
)
def test_benches_w1299(fam):
    out = getattr(benches_w1299, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
