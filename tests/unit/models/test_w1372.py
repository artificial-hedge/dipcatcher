import pytest


@pytest.mark.parametrize(
    "name",
    [
        "billsum_lite_studies",
        "govreport_lite_studies",
        "elm_lite_studies",
        "qmsum_lite_studies",
        "wikisum_lite_studies",
        "booksum_lite_studies",
    ],
)
def test_w1372_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
