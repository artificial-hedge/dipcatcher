import pytest


@pytest.mark.parametrize(
    "name",
    [
        "agent_memory_studies",
        "code_agent_studies",
        "computer_use_studies",
        "mcp_protocol_studies",
        "skill_library_studies",
        "web_agent_studies",
    ],
)
def test_w1274_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
