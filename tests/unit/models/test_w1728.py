import pytest


@pytest.mark.parametrize(
    "name",
    [
        "isten_qa_studies",
        "csaba_qa_studies",
        "garabonci_qa_studies",
        "liderc_qa_studies",
        "boszorka_qa_studies",
        "taltos_qa_studies",
    ],
)
def test_w1728_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
