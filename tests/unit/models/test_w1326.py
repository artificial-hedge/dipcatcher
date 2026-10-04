import pytest


@pytest.mark.parametrize(
    "name",
    [
        "membership_inference_studies",
        "attribute_inference_studies",
        "extraction_attack_studies",
        "model_inversion_studies",
        "canary_memorization_studies",
        "privacy_meter_studies",
    ],
)
def test_w1326_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
