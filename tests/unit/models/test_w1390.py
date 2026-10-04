import pytest


@pytest.mark.parametrize(
    "name",
    [
        "episum_lite_studies",
        "sqcs_lite_studies",
        "mds_news_studies",
        "summon_fce_studies",
        "wcep_lite_studies",
        "fsum_lite_studies",
    ],
)
def test_w1390_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
