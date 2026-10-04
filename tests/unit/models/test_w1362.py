import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fake_news_studies",
        "emergent_lite_studies",
        "stance_detect_studies",
        "check_that_studies",
        "claim_buster_studies",
        "snopes_lite_studies",
    ],
)
def test_w1362_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
