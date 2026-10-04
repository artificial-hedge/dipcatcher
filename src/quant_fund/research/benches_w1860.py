"""Wave-1860 bench adapters: arthurian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    arthur_qa_studies,
    elaine_qa_studies,
    gorlois_qa_studies,
    igraine_qa_studies,
    morgan_qa_studies,
    vivien_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arthur_qa_studies_family(seed: int = _SEED + 0):
    """arthur_qa_studies: synthetic correctness bench."""
    return _finite_blob(arthur_qa_studies.bench_arthur_qa_studies(seed))


def bench_elaine_qa_studies_family(seed: int = _SEED + 1):
    """elaine_qa_studies: synthetic correctness bench."""
    return _finite_blob(elaine_qa_studies.bench_elaine_qa_studies(seed))


def bench_gorlois_qa_studies_family(seed: int = _SEED + 2):
    """gorlois_qa_studies: synthetic correctness bench."""
    return _finite_blob(gorlois_qa_studies.bench_gorlois_qa_studies(seed))


def bench_igraine_qa_studies_family(seed: int = _SEED + 3):
    """igraine_qa_studies: synthetic correctness bench."""
    return _finite_blob(igraine_qa_studies.bench_igraine_qa_studies(seed))


def bench_morgan_qa_studies_family(seed: int = _SEED + 4):
    """morgan_qa_studies: synthetic correctness bench."""
    return _finite_blob(morgan_qa_studies.bench_morgan_qa_studies(seed))


def bench_vivien_qa_studies_family(seed: int = _SEED + 5):
    """vivien_qa_studies: synthetic correctness bench."""
    return _finite_blob(vivien_qa_studies.bench_vivien_qa_studies(seed))
