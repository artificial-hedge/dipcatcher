import pytest


@pytest.mark.parametrize(
    "name",
    [
        "div_demon_qa_studies",
        "nasu_demon_qa_studies",
        "druj_spirit_qa_studies",
        "div_aq_qa_studies",
        "demon_div_qa_studies",
        "yalburz_qa_studies",
    ],
)
def test_w1910_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
