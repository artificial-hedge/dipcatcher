import pytest


@pytest.mark.parametrize(
    "name",
    [
        "audit_qa_studies",
        "broker_qa_studies",
        "bank_qa_studies",
        "credit_qa_studies",
        "earnings_qa_studies",
        "analyst_qa_studies",
    ],
)
def test_w1416_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
