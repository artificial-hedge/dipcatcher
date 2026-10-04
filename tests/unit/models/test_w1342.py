import pytest


@pytest.mark.parametrize(
    "name",
    [
        "regard_metric_studies",
        "gender_bias_studies",
        "nlp_bias_studies",
        "fairness_eval_studies",
        "jigsaw_tox_studies",
        "pronoun_bias_studies",
    ],
)
def test_w1342_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
