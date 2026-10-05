"""Wave-1953 bench adapters: goetic-ordinance canon (SYNTHETIC only)."""

from quant_fund.models import (
    amy_qa_studies,
    andrealphus_qa_studies,
    gremory_qa_studies,
    murmur_qa_studies,
    orobas_qa_studies,
    ose_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19530


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amy_qa_studies_family(seed: int = _SEED + 0):
    """amy_qa_studies: synthetic correctness bench."""
    return _finite_blob(amy_qa_studies.bench_amy_qa_studies(seed))



def bench_andrealphus_qa_studies_family(seed: int = _SEED + 1):
    """andrealphus_qa_studies: synthetic correctness bench."""
    return _finite_blob(andrealphus_qa_studies.bench_andrealphus_qa_studies(seed))



def bench_gremory_qa_studies_family(seed: int = _SEED + 2):
    """gremory_qa_studies: synthetic correctness bench."""
    return _finite_blob(gremory_qa_studies.bench_gremory_qa_studies(seed))



def bench_murmur_qa_studies_family(seed: int = _SEED + 3):
    """murmur_qa_studies: synthetic correctness bench."""
    return _finite_blob(murmur_qa_studies.bench_murmur_qa_studies(seed))



def bench_orobas_qa_studies_family(seed: int = _SEED + 4):
    """orobas_qa_studies: synthetic correctness bench."""
    return _finite_blob(orobas_qa_studies.bench_orobas_qa_studies(seed))



def bench_ose_qa_studies_family(seed: int = _SEED + 5):
    """ose_qa_studies: synthetic correctness bench."""
    return _finite_blob(ose_qa_studies.bench_ose_qa_studies(seed))
