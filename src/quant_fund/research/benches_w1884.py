"""Wave-1884 bench adapters: saharan-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    argemm_qa_studies,
    arzew_qa_studies,
    cilteni_qa_studies,
    essuf_qa_studies,
    medghassen_qa_studies,
    tanezruft_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18840


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_argemm_qa_studies_family(seed: int = _SEED + 0):
    """argemm_qa_studies: synthetic correctness bench."""
    return _finite_blob(argemm_qa_studies.bench_argemm_qa_studies(seed))


def bench_arzew_qa_studies_family(seed: int = _SEED + 1):
    """arzew_qa_studies: synthetic correctness bench."""
    return _finite_blob(arzew_qa_studies.bench_arzew_qa_studies(seed))


def bench_cilteni_qa_studies_family(seed: int = _SEED + 2):
    """cilteni_qa_studies: synthetic correctness bench."""
    return _finite_blob(cilteni_qa_studies.bench_cilteni_qa_studies(seed))


def bench_essuf_qa_studies_family(seed: int = _SEED + 3):
    """essuf_qa_studies: synthetic correctness bench."""
    return _finite_blob(essuf_qa_studies.bench_essuf_qa_studies(seed))


def bench_medghassen_qa_studies_family(seed: int = _SEED + 4):
    """medghassen_qa_studies: synthetic correctness bench."""
    return _finite_blob(medghassen_qa_studies.bench_medghassen_qa_studies(seed))


def bench_tanezruft_qa_studies_family(seed: int = _SEED + 5):
    """tanezruft_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanezruft_qa_studies.bench_tanezruft_qa_studies(seed))
