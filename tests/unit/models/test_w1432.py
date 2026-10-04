import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cuisine_qa_studies",
        "dish_qa_studies",
        "dessert_qa_studies",
        "fruit_qa_studies",
        "ingredient_qa_studies",
        "beverage_qa_studies",
    ],
)
def test_w1432_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
