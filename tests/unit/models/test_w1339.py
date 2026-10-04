import pytest


@pytest.mark.parametrize(
    "name",
    [
        "math_odyssey_studies",
        "hol_step_studies",
        "tab_math_studies",
        "geo_qa_studies",
        "uni_math_studies",
        "aqua_rat_studies",
    ],
)
def test_w1339_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
