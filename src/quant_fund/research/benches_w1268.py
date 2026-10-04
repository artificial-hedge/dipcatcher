"""Wave-1268 bench adapters: RL-skills/goal canon (SYNTHETIC only)."""

from quant_fund.models import (
    curiosity_diversity_studies,
    hindsight_relabel_studies,
    occupancy_measure_studies,
    option_discovery_studies,
    skill_chain_studies,
    successor_feature_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12680


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_curiosity_diversity_studies_family(seed: int = _SEED + 0):
    """curiosity_diversity_studies: synthetic correctness bench."""
    return _finite_blob(curiosity_diversity_studies.bench_curiosity_diversity_studies(seed))


def bench_hindsight_relabel_studies_family(seed: int = _SEED + 1):
    """hindsight_relabel_studies: synthetic correctness bench."""
    return _finite_blob(hindsight_relabel_studies.bench_hindsight_relabel_studies(seed))


def bench_occupancy_measure_studies_family(seed: int = _SEED + 2):
    """occupancy_measure_studies: synthetic correctness bench."""
    return _finite_blob(occupancy_measure_studies.bench_occupancy_measure_studies(seed))


def bench_option_discovery_studies_family(seed: int = _SEED + 3):
    """option_discovery_studies: synthetic correctness bench."""
    return _finite_blob(option_discovery_studies.bench_option_discovery_studies(seed))


def bench_skill_chain_studies_family(seed: int = _SEED + 4):
    """skill_chain_studies: synthetic correctness bench."""
    return _finite_blob(skill_chain_studies.bench_skill_chain_studies(seed))


def bench_successor_feature_studies_family(seed: int = _SEED + 5):
    """successor_feature_studies: synthetic correctness bench."""
    return _finite_blob(successor_feature_studies.bench_successor_feature_studies(seed))
