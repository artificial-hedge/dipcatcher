import pytest


@pytest.mark.parametrize(
    "name",
    [
        "citation_check_studies",
        "claim_verifier_studies",
        "entailment_studies",
        "factuality_score_studies",
        "grounding_verify_studies",
        "self_reflect_studies",
    ],
)
def test_w1284_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
