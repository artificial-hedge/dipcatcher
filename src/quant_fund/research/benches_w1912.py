"""Wave-1912 bench adapters: slavic-demon-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    belun_qa_studies,
    indrik_qa_studies,
    koshchey_qa_studies,
    mavka_qa_studies,
    psoglav_qa_studies,
    triglav_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19120


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_belun_qa_studies_family(seed: int = _SEED + 0):
    """belun_qa_studies: synthetic correctness bench."""
    return _finite_blob(belun_qa_studies.bench_belun_qa_studies(seed))


def bench_indrik_qa_studies_family(seed: int = _SEED + 1):
    """indrik_qa_studies: synthetic correctness bench."""
    return _finite_blob(indrik_qa_studies.bench_indrik_qa_studies(seed))


def bench_koshchey_qa_studies_family(seed: int = _SEED + 2):
    """koshchey_qa_studies: synthetic correctness bench."""
    return _finite_blob(koshchey_qa_studies.bench_koshchey_qa_studies(seed))


def bench_mavka_qa_studies_family(seed: int = _SEED + 3):
    """mavka_qa_studies: synthetic correctness bench."""
    return _finite_blob(mavka_qa_studies.bench_mavka_qa_studies(seed))


def bench_psoglav_qa_studies_family(seed: int = _SEED + 4):
    """psoglav_qa_studies: synthetic correctness bench."""
    return _finite_blob(psoglav_qa_studies.bench_psoglav_qa_studies(seed))


def bench_triglav_qa_studies_family(seed: int = _SEED + 5):
    """triglav_qa_studies: synthetic correctness bench."""
    return _finite_blob(triglav_qa_studies.bench_triglav_qa_studies(seed))
