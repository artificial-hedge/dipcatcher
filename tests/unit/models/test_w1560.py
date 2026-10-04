import pytest


@pytest.mark.parametrize(
    "name",
    [
        "damselfish_qa_studies",
        "butterflyfish_qa_studies",
        "wrasse_qa_studies",
        "parrotfish_qa_studies",
        "grouper_qa_studies",
        "snapper_qa_studies",
    ],
)
def test_w1560_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
