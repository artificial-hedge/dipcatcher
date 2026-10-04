import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bel2_qa_studies",
        "yarhibol2_qa_studies",
        "aglibol2_qa_studies",
        "malakbel2_qa_studies",
        "baalshamin2_qa_studies",
        "astarte2_qa_studies",
    ],
)
def test_w1838_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
