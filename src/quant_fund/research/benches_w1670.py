"""Wave-1670 bench adapters: greek-spirit canon (SYNTHETIC only)."""

from quant_fund.models import (
    alseid_qa_studies,
    gnome_volk_qa_studies,
    meliae_qa_studies,
    napaea_qa_studies,
    oread_qa_studies,
    sylph_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16700


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alseid_qa_studies_family(seed: int = _SEED + 0):
    """alseid_qa_studies: synthetic correctness bench."""
    return _finite_blob(alseid_qa_studies.bench_alseid_qa_studies(seed))


def bench_gnome_volk_qa_studies_family(seed: int = _SEED + 1):
    """gnome_volk_qa_studies: synthetic correctness bench."""
    return _finite_blob(gnome_volk_qa_studies.bench_gnome_volk_qa_studies(seed))


def bench_meliae_qa_studies_family(seed: int = _SEED + 2):
    """meliae_qa_studies: synthetic correctness bench."""
    return _finite_blob(meliae_qa_studies.bench_meliae_qa_studies(seed))


def bench_napaea_qa_studies_family(seed: int = _SEED + 3):
    """napaea_qa_studies: synthetic correctness bench."""
    return _finite_blob(napaea_qa_studies.bench_napaea_qa_studies(seed))


def bench_oread_qa_studies_family(seed: int = _SEED + 4):
    """oread_qa_studies: synthetic correctness bench."""
    return _finite_blob(oread_qa_studies.bench_oread_qa_studies(seed))


def bench_sylph_qa_studies_family(seed: int = _SEED + 5):
    """sylph_qa_studies: synthetic correctness bench."""
    return _finite_blob(sylph_qa_studies.bench_sylph_qa_studies(seed))
