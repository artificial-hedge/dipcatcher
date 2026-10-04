import pytest


@pytest.mark.parametrize(
    "name",
    [
        "parsley_qa_studies",
        "tarragon_qa_studies",
        "rosemary_qa_studies",
        "saffron_qa_studies",
        "turmeric_qa_studies",
        "oregano_qa_studies",
    ],
)
def test_w1499_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
