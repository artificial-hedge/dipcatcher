import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vedmak_qa_studies",
        "kladenets_qa_studies",
        "leshii_qa_studies",
        "morozko_qa_studies",
        "kostroma_qa_studies",
        "yarilo_qa_studies",
    ],
)
def test_w1690_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
