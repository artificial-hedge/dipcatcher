"""Wave-1634 bench adapters: elemental-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    gnome_2_qa_studies,
    ifrit_qa_studies,
    marid_qa_studies,
    salamander_2_qa_studies,
    sylph_2_qa_studies,
    undine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16340


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_gnome_2_qa_studies_family(seed: int = _SEED + 0):
    """gnome_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(gnome_2_qa_studies.bench_gnome_2_qa_studies(seed))


def bench_ifrit_qa_studies_family(seed: int = _SEED + 1):
    """ifrit_qa_studies: synthetic correctness bench."""
    return _finite_blob(ifrit_qa_studies.bench_ifrit_qa_studies(seed))


def bench_marid_qa_studies_family(seed: int = _SEED + 2):
    """marid_qa_studies: synthetic correctness bench."""
    return _finite_blob(marid_qa_studies.bench_marid_qa_studies(seed))


def bench_salamander_2_qa_studies_family(seed: int = _SEED + 3):
    """salamander_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(salamander_2_qa_studies.bench_salamander_2_qa_studies(seed))


def bench_sylph_2_qa_studies_family(seed: int = _SEED + 4):
    """sylph_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sylph_2_qa_studies.bench_sylph_2_qa_studies(seed))


def bench_undine_qa_studies_family(seed: int = _SEED + 5):
    """undine_qa_studies: synthetic correctness bench."""
    return _finite_blob(undine_qa_studies.bench_undine_qa_studies(seed))
