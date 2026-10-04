import pytest


@pytest.mark.parametrize(
    "name",
    [
        "self_bleu_studies",
        "diversity_eval_studies",
        "factscore_studies",
        "attribution_eval_studies",
        "citation_eval_studies",
        "alpaca_eval_studies",
    ],
)
def test_w1310_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
