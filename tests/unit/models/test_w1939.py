import pytest


@pytest.mark.parametrize(
    "name",
    [
        "phi_pret_qa_studies",
        "phi_taen_qa_studies",
        "phi_khao_qa_studies",
        "phi_puay_qa_studies",
        "phi_rai_qa_studies",
        "phi_yuan_qa_studies",
    ],
)
def test_w1939_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
