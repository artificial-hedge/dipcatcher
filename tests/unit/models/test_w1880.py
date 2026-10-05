import pytest


@pytest.mark.parametrize(
    "name",
    [
        "milkart_qa_studies",
        "achimi_qa_studies",
        "tamgak_qa_studies",
        "tesfit_qa_studies",
        "iyezid_qa_studies",
        "mazer_qa_studies",
    ],
)
def test_w1880_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
