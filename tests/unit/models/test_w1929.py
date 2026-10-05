import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tipua_qa_studies",
        "nuku_mai_tore_qa_studies",
        "kahui_tipua_qa_studies",
        "wheke_muturangi_qa_studies",
        "hotupuku_qa_studies",
        "kataore_qa_studies",
    ],
)
def test_w1929_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
