import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hendrycks_test_studies",
        "hotpot_lite_studies",
        "squad_lite2_studies",
        "multirc_lite_studies",
        "record_lite_studies",
        "quoref_lite_studies",
    ],
)
def test_w1357_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
