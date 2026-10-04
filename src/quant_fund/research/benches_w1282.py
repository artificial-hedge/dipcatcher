"""Wave-1282 bench adapters: embodied-VLA canon (SYNTHETIC only)."""

from quant_fund.models import (
    affordance_map_studies,
    embodied_agent_studies,
    spatial_reasoning_studies,
    video_diffusion_studies,
    vla_model_studies,
    world_sim_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12820


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_affordance_map_studies_family(seed: int = _SEED + 0):
    """affordance_map_studies: synthetic correctness bench."""
    return _finite_blob(affordance_map_studies.bench_affordance_map_studies(seed))


def bench_embodied_agent_studies_family(seed: int = _SEED + 1):
    """embodied_agent_studies: synthetic correctness bench."""
    return _finite_blob(embodied_agent_studies.bench_embodied_agent_studies(seed))


def bench_spatial_reasoning_studies_family(seed: int = _SEED + 2):
    """spatial_reasoning_studies: synthetic correctness bench."""
    return _finite_blob(spatial_reasoning_studies.bench_spatial_reasoning_studies(seed))


def bench_video_diffusion_studies_family(seed: int = _SEED + 3):
    """video_diffusion_studies: synthetic correctness bench."""
    return _finite_blob(video_diffusion_studies.bench_video_diffusion_studies(seed))


def bench_vla_model_studies_family(seed: int = _SEED + 4):
    """vla_model_studies: synthetic correctness bench."""
    return _finite_blob(vla_model_studies.bench_vla_model_studies(seed))


def bench_world_sim_studies_family(seed: int = _SEED + 5):
    """world_sim_studies: synthetic correctness bench."""
    return _finite_blob(world_sim_studies.bench_world_sim_studies(seed))
