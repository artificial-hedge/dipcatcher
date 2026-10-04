import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aime_eval_studies",
        "mgsm_lite_studies",
        "math500_lite_studies",
        "minerva_math_studies",
        "svamp_lite_studies",
        "asdiv_lite_studies",
    ],
)
def test_w1369_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
