import pytest


@pytest.mark.parametrize(
    "name",
    [
        "clause_qa_studies",
        "lawqa_lite_studies",
        "contract_qa_studies",
        "legal_qa_studies",
        "statute_qa_studies",
        "case_qa_studies",
    ],
)
def test_w1415_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
