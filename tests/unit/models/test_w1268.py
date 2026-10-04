import pytest


@pytest.mark.parametrize(
    "name",
    [
        "curiosity_diversity_studies",
        "hindsight_relabel_studies",
        "occupancy_measure_studies",
        "option_discovery_studies",
        "skill_chain_studies",
        "successor_feature_studies",
    ],
)
def test_w1268_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
