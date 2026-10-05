import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bahamut_qa_studies",
        "falak_qa_studies",
        "karkadann_qa_studies",
        "nasnas_qa_studies",
        "shahmaran_qa_studies",
        "ghoula_qa_studies",
    ],
)
def test_w1886_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
