import pytest


@pytest.mark.parametrize(
    "name",
    [
        "toxigen_eval_studies",
        "bbq_bias_studies",
        "crowspairs_studies",
        "holist_bias_studies",
        "bold_bias_studies",
        "realtoxicity_studies",
    ],
)
def test_w1327_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
