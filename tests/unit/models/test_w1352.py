import pytest


@pytest.mark.parametrize(
    "name",
    [
        "social_iqa2_studies",
        "wino_bias_studies",
        "stereo_lite_studies",
        "crowsp_lite_studies",
        "bias_bench_studies",
        "honesty_lie_studies",
    ],
)
def test_w1352_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
