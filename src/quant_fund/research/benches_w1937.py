"""Wave-1937 bench adapters: chiloe-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    caleuche_qa_studies,
    camahueto_qa_studies,
    fiura_qa_studies,
    invunche_qa_studies,
    pincoya_qa_studies,
    trauco_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19370


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_caleuche_qa_studies_family(seed: int = _SEED + 0):
    """caleuche_qa_studies: synthetic correctness bench."""
    return _finite_blob(caleuche_qa_studies.bench_caleuche_qa_studies(seed))


def bench_camahueto_qa_studies_family(seed: int = _SEED + 1):
    """camahueto_qa_studies: synthetic correctness bench."""
    return _finite_blob(camahueto_qa_studies.bench_camahueto_qa_studies(seed))


def bench_fiura_qa_studies_family(seed: int = _SEED + 2):
    """fiura_qa_studies: synthetic correctness bench."""
    return _finite_blob(fiura_qa_studies.bench_fiura_qa_studies(seed))


def bench_invunche_qa_studies_family(seed: int = _SEED + 3):
    """invunche_qa_studies: synthetic correctness bench."""
    return _finite_blob(invunche_qa_studies.bench_invunche_qa_studies(seed))


def bench_pincoya_qa_studies_family(seed: int = _SEED + 4):
    """pincoya_qa_studies: synthetic correctness bench."""
    return _finite_blob(pincoya_qa_studies.bench_pincoya_qa_studies(seed))


def bench_trauco_qa_studies_family(seed: int = _SEED + 5):
    """trauco_qa_studies: synthetic correctness bench."""
    return _finite_blob(trauco_qa_studies.bench_trauco_qa_studies(seed))
