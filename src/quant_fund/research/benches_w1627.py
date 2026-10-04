"""Wave-1627 bench adapters: cave-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cave_crayfish_qa_studies,
    cave_scorpion_qa_studies,
    cave_springtail_qa_studies,
    cave_worm_qa_studies,
    stygobite_qa_studies,
    troglofish_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16270


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cave_crayfish_qa_studies_family(seed: int = _SEED + 0):
    """cave_crayfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_crayfish_qa_studies.bench_cave_crayfish_qa_studies(seed))


def bench_cave_scorpion_qa_studies_family(seed: int = _SEED + 1):
    """cave_scorpion_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_scorpion_qa_studies.bench_cave_scorpion_qa_studies(seed))


def bench_cave_springtail_qa_studies_family(seed: int = _SEED + 2):
    """cave_springtail_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_springtail_qa_studies.bench_cave_springtail_qa_studies(seed))


def bench_cave_worm_qa_studies_family(seed: int = _SEED + 3):
    """cave_worm_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_worm_qa_studies.bench_cave_worm_qa_studies(seed))


def bench_stygobite_qa_studies_family(seed: int = _SEED + 4):
    """stygobite_qa_studies: synthetic correctness bench."""
    return _finite_blob(stygobite_qa_studies.bench_stygobite_qa_studies(seed))


def bench_troglofish_qa_studies_family(seed: int = _SEED + 5):
    """troglofish_qa_studies: synthetic correctness bench."""
    return _finite_blob(troglofish_qa_studies.bench_troglofish_qa_studies(seed))
