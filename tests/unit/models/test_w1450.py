import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cow_qa_studies",
        "horse_qa_studies",
        "goat_qa_studies",
        "pig_qa_studies",
        "sheep_qa_studies",
        "barn_qa_studies",
    ],
)
def test_w1450_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
