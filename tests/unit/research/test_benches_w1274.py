import pytest

from quant_fund.research import benches_w1274


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agent_memory_studies_family",
        "bench_code_agent_studies_family",
        "bench_computer_use_studies_family",
        "bench_mcp_protocol_studies_family",
        "bench_skill_library_studies_family",
        "bench_web_agent_studies_family",
    ],
)
def test_benches_w1274(fam):
    out = getattr(benches_w1274, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
