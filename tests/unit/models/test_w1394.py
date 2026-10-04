import pytest


@pytest.mark.parametrize(
    "name",
    [
        "apps_lite_studies",
        "mbpp_lite_studies",
        "livecode_studies",
        "restbench_studies",
        "swe_gym_studies",
        "api_eval_studies",
    ],
)
def test_w1394_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
