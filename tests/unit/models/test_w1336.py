import pytest


@pytest.mark.parametrize(
    "name",
    [
        "web_nav_studies",
        "mind2web_studies",
        "visual_web_studies",
        "miniwob_studies",
        "webarena_studies",
        "assistantbench_studies",
    ],
)
def test_w1336_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
