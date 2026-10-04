import pytest

from quant_fund.research import benches_w1282


@pytest.mark.parametrize(
    "fam",
    [
        "bench_affordance_map_studies_family",
        "bench_embodied_agent_studies_family",
        "bench_spatial_reasoning_studies_family",
        "bench_video_diffusion_studies_family",
        "bench_vla_model_studies_family",
        "bench_world_sim_studies_family",
    ],
)
def test_benches_w1282(fam):
    out = getattr(benches_w1282, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
