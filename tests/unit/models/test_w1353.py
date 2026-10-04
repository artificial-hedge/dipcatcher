import pytest


@pytest.mark.parametrize(
    "name",
    [
        "strategy_qa_studies",
        "entailment_bn_studies",
        "creak_lite_studies",
        "prove_it_studies",
        "sup_nli_studies",
        "hans_lite_studies",
    ],
)
def test_w1353_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
