import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cherry_qa_studies",
        "lemon_qa_studies",
        "grape_qa_studies",
        "mango_qa_studies",
        "peach_qa_studies",
        "apple_qa_studies",
    ],
)
def test_w1451_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
