import pytest


@pytest.mark.parametrize(
    "name",
    [
        "activation_patch_studies",
        "circuit_tracer_studies",
        "feature_dashboard_studies",
        "jailbreak_detect_studies",
        "mech_anomaly_studies",
        "sae_linter_studies",
    ],
)
def test_w1295_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
