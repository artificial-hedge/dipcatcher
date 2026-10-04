import pytest


@pytest.mark.parametrize(
    "name",
    [
        "attribution_graph_studies",
        "causal_tracing_studies",
        "circuit_discovery_studies",
        "feature_geometry_studies",
        "gated_sae_studies",
        "transcoder_studies",
    ],
)
def test_w1270_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
