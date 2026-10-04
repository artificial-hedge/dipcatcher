import pytest

from quant_fund.research import benches_w1287


@pytest.mark.parametrize(
    "fam",
    [
        "bench_capability_eval_studies_family",
        "bench_control_eval_studies_family",
        "bench_deception_eval_studies_family",
        "bench_prompt_injection_studies_family",
        "bench_sandbox_escape_studies_family",
        "bench_tool_call_verify_studies_family",
    ],
)
def test_benches_w1287(fam):
    out = getattr(benches_w1287, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
