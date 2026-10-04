import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nest_tools_studies",
        "toolqa_lite_studies",
        "toolbench2_studies",
        "ultra_tool_studies",
        "work_plus_studies",
        "meta_tool_studies",
    ],
)
def test_w1392_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
