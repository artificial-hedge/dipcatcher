"""Wave-1810 bench adapters: yoruba-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    elegba2_qa_studies,
    obatala2_qa_studies,
    orunmila2_qa_studies,
    osun2_qa_studies,
    oya2_qa_studies,
    shango2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18100


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_elegba2_qa_studies_family(seed: int = _SEED + 0):
    """elegba2_qa_studies: synthetic correctness bench."""
    return _finite_blob(elegba2_qa_studies.bench_elegba2_qa_studies(seed))


def bench_obatala2_qa_studies_family(seed: int = _SEED + 1):
    """obatala2_qa_studies: synthetic correctness bench."""
    return _finite_blob(obatala2_qa_studies.bench_obatala2_qa_studies(seed))


def bench_orunmila2_qa_studies_family(seed: int = _SEED + 2):
    """orunmila2_qa_studies: synthetic correctness bench."""
    return _finite_blob(orunmila2_qa_studies.bench_orunmila2_qa_studies(seed))


def bench_osun2_qa_studies_family(seed: int = _SEED + 3):
    """osun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(osun2_qa_studies.bench_osun2_qa_studies(seed))


def bench_oya2_qa_studies_family(seed: int = _SEED + 4):
    """oya2_qa_studies: synthetic correctness bench."""
    return _finite_blob(oya2_qa_studies.bench_oya2_qa_studies(seed))


def bench_shango2_qa_studies_family(seed: int = _SEED + 5):
    """shango2_qa_studies: synthetic correctness bench."""
    return _finite_blob(shango2_qa_studies.bench_shango2_qa_studies(seed))
