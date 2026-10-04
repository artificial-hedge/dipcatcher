import pytest


@pytest.mark.parametrize(
    "name",
    [
        "begins_lite_studies",
        "multi_woz_studies",
        "faithful_dial_studies",
        "top_dialog_studies",
        "wow_lite_studies",
        "diamonds_lite_studies",
    ],
)
def test_w1381_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
