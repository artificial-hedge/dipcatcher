import pytest


@pytest.mark.parametrize(
    "name",
    [
        "analogy_qa_studies",
        "entailment_qa_studies",
        "arct_lite_studies",
        "fusion_qa_studies",
        "proof_qa_studies",
        "abduct_qa_studies",
    ],
)
def test_w1408_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
