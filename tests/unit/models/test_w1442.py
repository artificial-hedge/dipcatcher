import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bee_qa_studies",
        "butterfly_qa_studies",
        "beetle_qa_studies",
        "cricket_qa_studies",
        "moth_qa_studies",
        "ant_qa_studies",
    ],
)
def test_w1442_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
