"""Wave-1288 bench adapters: RL-imitation canon (SYNTHETIC only)."""

from quant_fund.models import (
    adversarial_irl_studies,
    behavior_cloning_studies,
    dagger_studies,
    offline_distill_studies,
    preference_irl_studies,
    skill_extraction_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12880


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adversarial_irl_studies_family(seed: int = _SEED + 0):
    """adversarial_irl_studies: synthetic correctness bench."""
    return _finite_blob(adversarial_irl_studies.bench_adversarial_irl_studies(seed))


def bench_behavior_cloning_studies_family(seed: int = _SEED + 1):
    """behavior_cloning_studies: synthetic correctness bench."""
    return _finite_blob(behavior_cloning_studies.bench_behavior_cloning_studies(seed))


def bench_dagger_studies_family(seed: int = _SEED + 2):
    """dagger_studies: synthetic correctness bench."""
    return _finite_blob(dagger_studies.bench_dagger_studies(seed))


def bench_offline_distill_studies_family(seed: int = _SEED + 3):
    """offline_distill_studies: synthetic correctness bench."""
    return _finite_blob(offline_distill_studies.bench_offline_distill_studies(seed))


def bench_preference_irl_studies_family(seed: int = _SEED + 4):
    """preference_irl_studies: synthetic correctness bench."""
    return _finite_blob(preference_irl_studies.bench_preference_irl_studies(seed))


def bench_skill_extraction_studies_family(seed: int = _SEED + 5):
    """skill_extraction_studies: synthetic correctness bench."""
    return _finite_blob(skill_extraction_studies.bench_skill_extraction_studies(seed))
