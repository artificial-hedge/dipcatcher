"""Wave-1849 bench adapters: aksumite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    alouq_qa_studies,
    aster2_qa_studies,
    beher_qa_studies,
    mahrem_qa_studies,
    medr_qa_studies,
    ruda_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18490


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alouq_qa_studies_family(seed: int = _SEED + 0):
    """alouq_qa_studies: synthetic correctness bench."""
    return _finite_blob(alouq_qa_studies.bench_alouq_qa_studies(seed))


def bench_aster2_qa_studies_family(seed: int = _SEED + 1):
    """aster2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aster2_qa_studies.bench_aster2_qa_studies(seed))


def bench_beher_qa_studies_family(seed: int = _SEED + 2):
    """beher_qa_studies: synthetic correctness bench."""
    return _finite_blob(beher_qa_studies.bench_beher_qa_studies(seed))


def bench_mahrem_qa_studies_family(seed: int = _SEED + 3):
    """mahrem_qa_studies: synthetic correctness bench."""
    return _finite_blob(mahrem_qa_studies.bench_mahrem_qa_studies(seed))


def bench_medr_qa_studies_family(seed: int = _SEED + 4):
    """medr_qa_studies: synthetic correctness bench."""
    return _finite_blob(medr_qa_studies.bench_medr_qa_studies(seed))


def bench_ruda_qa_studies_family(seed: int = _SEED + 5):
    """ruda_qa_studies: synthetic correctness bench."""
    return _finite_blob(ruda_qa_studies.bench_ruda_qa_studies(seed))
