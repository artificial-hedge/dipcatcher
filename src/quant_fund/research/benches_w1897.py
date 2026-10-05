"""Wave-1897 bench adapters: european-vampire canon (SYNTHETIC only)."""

from quant_fund.models import (
    lamia_qa_studies,
    mormo_qa_studies,
    nachzehrer_qa_studies,
    striga_qa_studies,
    strix_qa_studies,
    vrykolakas_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18970


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_lamia_qa_studies_family(seed: int = _SEED + 0):
    """lamia_qa_studies: synthetic correctness bench."""
    return _finite_blob(lamia_qa_studies.bench_lamia_qa_studies(seed))


def bench_mormo_qa_studies_family(seed: int = _SEED + 1):
    """mormo_qa_studies: synthetic correctness bench."""
    return _finite_blob(mormo_qa_studies.bench_mormo_qa_studies(seed))


def bench_nachzehrer_qa_studies_family(seed: int = _SEED + 2):
    """nachzehrer_qa_studies: synthetic correctness bench."""
    return _finite_blob(nachzehrer_qa_studies.bench_nachzehrer_qa_studies(seed))


def bench_striga_qa_studies_family(seed: int = _SEED + 3):
    """striga_qa_studies: synthetic correctness bench."""
    return _finite_blob(striga_qa_studies.bench_striga_qa_studies(seed))


def bench_strix_qa_studies_family(seed: int = _SEED + 4):
    """strix_qa_studies: synthetic correctness bench."""
    return _finite_blob(strix_qa_studies.bench_strix_qa_studies(seed))


def bench_vrykolakas_qa_studies_family(seed: int = _SEED + 5):
    """vrykolakas_qa_studies: synthetic correctness bench."""
    return _finite_blob(vrykolakas_qa_studies.bench_vrykolakas_qa_studies(seed))
