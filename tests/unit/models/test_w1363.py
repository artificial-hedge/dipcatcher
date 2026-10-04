import pytest


@pytest.mark.parametrize(
    "name",
    [
        "age_bias_studies",
        "dialect_bias_studies",
        "cw_qa2_studies",
        "politi_fact_studies",
        "rumor_twitter_studies",
        "curry_qa_studies",
    ],
)
def test_w1363_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
