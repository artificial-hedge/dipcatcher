import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mcp_bench_studies",
        "tool_sandbox_studies",
        "net_hack_studies",
        "videoweb_studies",
        "webshop_lite_studies",
        "hamming_mcp_studies",
    ],
)
def test_w1393_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
