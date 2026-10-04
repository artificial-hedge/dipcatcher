import pytest


@pytest.mark.parametrize(
    "name",
    [
        "moonrat_qa_studies",
        "marsupial_mole_qa_studies",
        "monotreme_qa_studies",
        "desman_qa_studies",
        "sengi_qa_studies",
        "moles_lite_qa_studies",
    ],
)
def test_w1602_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
