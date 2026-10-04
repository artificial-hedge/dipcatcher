import pytest


@pytest.mark.parametrize(
    "name",
    [
        "skate_qa_studies",
        "wobbegong_qa_studies",
        "fiddler_qa_studies",
        "sea_snake_qa_studies",
        "decorator_qa_studies",
        "rock_crab_qa_studies",
    ],
)
def test_w1605_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
