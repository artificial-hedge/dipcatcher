import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tupilaq_qa_studies",
        "ijiraq_qa_studies",
        "mahaha_qa_studies",
        "qivittoq_qa_studies",
        "amautalik_qa_studies",
        "tornit_qa_studies",
    ],
)
def test_w1927_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
