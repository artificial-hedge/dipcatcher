import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lcmc_lite_studies",
        "scan_cfsp_studies",
        "mco_lite_studies",
        "hops_add_studies",
        "dyck_lang_studies",
        "shuffle_expr_studies",
    ],
)
def test_w1350_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
