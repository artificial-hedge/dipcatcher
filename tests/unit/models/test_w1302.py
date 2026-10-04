import pytest


@pytest.mark.parametrize(
    "name",
    [
        "membership_infer_studies",
        "shadow_model_studies",
        "lira_studies",
        "canary_infer_studies",
        "gradient_leak_studies",
        "deep_leak_studies",
    ],
)
def test_w1302_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
