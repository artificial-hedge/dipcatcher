"""Wave-1922 bench adapters: finnish-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    kalma_qa_studies,
    kratti_qa_studies,
    loviatar_qa_studies,
    nakki_qa_studies,
    painajainen_qa_studies,
    tursas_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19220


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_kalma_qa_studies_family(seed: int = _SEED + 0):
    """kalma_qa_studies: synthetic correctness bench."""
    return _finite_blob(kalma_qa_studies.bench_kalma_qa_studies(seed))


def bench_kratti_qa_studies_family(seed: int = _SEED + 1):
    """kratti_qa_studies: synthetic correctness bench."""
    return _finite_blob(kratti_qa_studies.bench_kratti_qa_studies(seed))


def bench_loviatar_qa_studies_family(seed: int = _SEED + 2):
    """loviatar_qa_studies: synthetic correctness bench."""
    return _finite_blob(loviatar_qa_studies.bench_loviatar_qa_studies(seed))


def bench_nakki_qa_studies_family(seed: int = _SEED + 3):
    """nakki_qa_studies: synthetic correctness bench."""
    return _finite_blob(nakki_qa_studies.bench_nakki_qa_studies(seed))


def bench_painajainen_qa_studies_family(seed: int = _SEED + 4):
    """painajainen_qa_studies: synthetic correctness bench."""
    return _finite_blob(painajainen_qa_studies.bench_painajainen_qa_studies(seed))


def bench_tursas_qa_studies_family(seed: int = _SEED + 5):
    """tursas_qa_studies: synthetic correctness bench."""
    return _finite_blob(tursas_qa_studies.bench_tursas_qa_studies(seed))
