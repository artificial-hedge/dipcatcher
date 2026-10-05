import pytest


@pytest.mark.parametrize(
    "name",
    [
        "enki2_qa_studies",
        "enlil2_qa_studies",
        "ninurta2_qa_studies",
        "namtar2_qa_studies",
        "gelal2_qa_studies",
        "zababa2_qa_studies",
    ],
)
def test_w1804_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
