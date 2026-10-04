import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cucumber_qa_studies",
        "onion_qa_studies",
        "garlic_qa_studies",
        "potato_qa_studies",
        "tomato_qa_studies",
        "carrot_qa_studies",
    ],
)
def test_w1452_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
