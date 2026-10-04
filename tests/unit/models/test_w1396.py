import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dialsum_lite_studies",
        "medsum_lite_studies",
        "facet_lite_studies",
        "oposum_lite_studies",
        "qsum_lite_studies",
        "agnews_lite_studies",
    ],
)
def test_w1396_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
