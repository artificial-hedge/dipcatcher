"""Wave-1934 bench adapters: mapuche-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    cherufe_qa_studies,
    chonchon_qa_studies,
    colo_colo_qa_studies,
    kalku_qa_studies,
    peuchen_qa_studies,
    wekufe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19340


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cherufe_qa_studies_family(seed: int = _SEED + 0):
    """cherufe_qa_studies: synthetic correctness bench."""
    return _finite_blob(cherufe_qa_studies.bench_cherufe_qa_studies(seed))



def bench_chonchon_qa_studies_family(seed: int = _SEED + 1):
    """chonchon_qa_studies: synthetic correctness bench."""
    return _finite_blob(chonchon_qa_studies.bench_chonchon_qa_studies(seed))



def bench_colo_colo_qa_studies_family(seed: int = _SEED + 2):
    """colo_colo_qa_studies: synthetic correctness bench."""
    return _finite_blob(colo_colo_qa_studies.bench_colo_colo_qa_studies(seed))



def bench_kalku_qa_studies_family(seed: int = _SEED + 3):
    """kalku_qa_studies: synthetic correctness bench."""
    return _finite_blob(kalku_qa_studies.bench_kalku_qa_studies(seed))



def bench_peuchen_qa_studies_family(seed: int = _SEED + 4):
    """peuchen_qa_studies: synthetic correctness bench."""
    return _finite_blob(peuchen_qa_studies.bench_peuchen_qa_studies(seed))



def bench_wekufe_qa_studies_family(seed: int = _SEED + 5):
    """wekufe_qa_studies: synthetic correctness bench."""
    return _finite_blob(wekufe_qa_studies.bench_wekufe_qa_studies(seed))
