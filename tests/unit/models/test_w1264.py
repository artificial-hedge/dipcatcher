import pytest


@pytest.mark.parametrize(
    "name",
    [
        "external_control_studies",
        "negative_control_studies",
        "probabilistic_bias_studies",
        "self_controlled_studies",
        "structural_nested_studies",
        "transportability_studies",
    ],
)
def test_w1264_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
