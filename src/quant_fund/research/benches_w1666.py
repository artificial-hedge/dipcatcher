"""Wave-1666 bench adapters: filipino-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    agta_qa_studies,
    berberoka_qa_studies,
    bungisngis_qa_studies,
    dalaketnon_qa_studies,
    ekek_qa_studies,
    engkanto_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16660


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agta_qa_studies_family(seed: int = _SEED + 0):
    """agta_qa_studies: synthetic correctness bench."""
    return _finite_blob(agta_qa_studies.bench_agta_qa_studies(seed))


def bench_berberoka_qa_studies_family(seed: int = _SEED + 1):
    """berberoka_qa_studies: synthetic correctness bench."""
    return _finite_blob(berberoka_qa_studies.bench_berberoka_qa_studies(seed))


def bench_bungisngis_qa_studies_family(seed: int = _SEED + 2):
    """bungisngis_qa_studies: synthetic correctness bench."""
    return _finite_blob(bungisngis_qa_studies.bench_bungisngis_qa_studies(seed))


def bench_dalaketnon_qa_studies_family(seed: int = _SEED + 3):
    """dalaketnon_qa_studies: synthetic correctness bench."""
    return _finite_blob(dalaketnon_qa_studies.bench_dalaketnon_qa_studies(seed))


def bench_ekek_qa_studies_family(seed: int = _SEED + 4):
    """ekek_qa_studies: synthetic correctness bench."""
    return _finite_blob(ekek_qa_studies.bench_ekek_qa_studies(seed))


def bench_engkanto_qa_studies_family(seed: int = _SEED + 5):
    """engkanto_qa_studies: synthetic correctness bench."""
    return _finite_blob(engkanto_qa_studies.bench_engkanto_qa_studies(seed))
