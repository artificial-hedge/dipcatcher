"""Wave-1569 bench adapters: ray canon (SYNTHETIC only)."""

from quant_fund.models import (
    eagle_ray_qa_studies,
    guitarfish_qa_studies,
    manta_qa_studies,
    sawfish_qa_studies,
    thornback_qa_studies,
    torpedo_ray_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15690


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_eagle_ray_qa_studies_family(seed: int = _SEED + 0):
    """eagle_ray_qa_studies: synthetic correctness bench."""
    return _finite_blob(eagle_ray_qa_studies.bench_eagle_ray_qa_studies(seed))


def bench_guitarfish_qa_studies_family(seed: int = _SEED + 1):
    """guitarfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(guitarfish_qa_studies.bench_guitarfish_qa_studies(seed))


def bench_manta_qa_studies_family(seed: int = _SEED + 2):
    """manta_qa_studies: synthetic correctness bench."""
    return _finite_blob(manta_qa_studies.bench_manta_qa_studies(seed))


def bench_sawfish_qa_studies_family(seed: int = _SEED + 3):
    """sawfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(sawfish_qa_studies.bench_sawfish_qa_studies(seed))


def bench_thornback_qa_studies_family(seed: int = _SEED + 4):
    """thornback_qa_studies: synthetic correctness bench."""
    return _finite_blob(thornback_qa_studies.bench_thornback_qa_studies(seed))


def bench_torpedo_ray_qa_studies_family(seed: int = _SEED + 5):
    """torpedo_ray_qa_studies: synthetic correctness bench."""
    return _finite_blob(torpedo_ray_qa_studies.bench_torpedo_ray_qa_studies(seed))
