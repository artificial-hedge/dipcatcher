"""Wave-1705 bench adapters: persian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ahriman_qa_studies,
    ahuramazda_qa_studies,
    anahita_qa_studies,
    mithra_qa_studies,
    verethragna_qa_studies,
    yazata_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17050


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ahriman_qa_studies_family(seed: int = _SEED + 0):
    """ahriman_qa_studies: synthetic correctness bench."""
    return _finite_blob(ahriman_qa_studies.bench_ahriman_qa_studies(seed))


def bench_ahuramazda_qa_studies_family(seed: int = _SEED + 1):
    """ahuramazda_qa_studies: synthetic correctness bench."""
    return _finite_blob(ahuramazda_qa_studies.bench_ahuramazda_qa_studies(seed))


def bench_anahita_qa_studies_family(seed: int = _SEED + 2):
    """anahita_qa_studies: synthetic correctness bench."""
    return _finite_blob(anahita_qa_studies.bench_anahita_qa_studies(seed))


def bench_mithra_qa_studies_family(seed: int = _SEED + 3):
    """mithra_qa_studies: synthetic correctness bench."""
    return _finite_blob(mithra_qa_studies.bench_mithra_qa_studies(seed))


def bench_verethragna_qa_studies_family(seed: int = _SEED + 4):
    """verethragna_qa_studies: synthetic correctness bench."""
    return _finite_blob(verethragna_qa_studies.bench_verethragna_qa_studies(seed))


def bench_yazata_qa_studies_family(seed: int = _SEED + 5):
    """yazata_qa_studies: synthetic correctness bench."""
    return _finite_blob(yazata_qa_studies.bench_yazata_qa_studies(seed))
