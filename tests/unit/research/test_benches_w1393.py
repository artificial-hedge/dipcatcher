import pytest

from quant_fund.research import benches_w1393


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hamming_mcp_studies_family",
        "bench_mcp_bench_studies_family",
        "bench_net_hack_studies_family",
        "bench_tool_sandbox_studies_family",
        "bench_videoweb_studies_family",
        "bench_webshop_lite_studies_family",
    ],
)
def test_benches_w1393(fam):
    out = getattr(benches_w1393, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
