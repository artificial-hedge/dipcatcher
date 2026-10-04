import pytest


@pytest.mark.parametrize(
    "name",
    [
        "alfworld_lite_studies",
        "jericho_lite_studies",
        "crafter_lite_studies",
        "scienceworld_studies",
        "textworld_lite_studies",
        "babyai_lite_studies",
    ],
)
def test_w1386_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
