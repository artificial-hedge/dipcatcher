import pytest


@pytest.mark.parametrize(
    "name",
    [
        "affordance_map_studies",
        "embodied_agent_studies",
        "spatial_reasoning_studies",
        "video_diffusion_studies",
        "vla_model_studies",
        "world_sim_studies",
    ],
)
def test_w1282_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
