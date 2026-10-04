import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ceramic_qa_studies",
        "iron_qa_studies",
        "glass_qa_studies",
        "steel_qa_studies",
        "wood_qa_studies",
        "alloy_qa_studies",
    ],
)
def test_w1434_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
