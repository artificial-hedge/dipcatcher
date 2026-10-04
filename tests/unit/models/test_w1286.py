import pytest


@pytest.mark.parametrize(
    "name",
    [
        "beacon_context_studies",
        "hierarchical_context_studies",
        "infini_attention_studies",
        "landmark_attention_studies",
        "ntk_scaling_studies",
        "yarn_scaling_studies",
    ],
)
def test_w1286_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
