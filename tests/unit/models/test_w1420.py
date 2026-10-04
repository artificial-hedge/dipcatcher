import pytest


@pytest.mark.parametrize(
    "name",
    [
        "claim_qa_studies",
        "deduction_qa_studies",
        "conclusion_qa_studies",
        "effect_qa_studies",
        "fallacy_qa_studies",
        "cause_qa_studies",
    ],
)
def test_w1420_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
