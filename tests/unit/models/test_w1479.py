import pytest


@pytest.mark.parametrize(
    "name",
    [
        "capybara_qa_studies",
        "tapir_qa_studies",
        "coati_qa_studies",
        "peccary_qa_studies",
        "agouti_qa_studies",
        "armadillo_qa_studies",
    ],
)
def test_w1479_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
