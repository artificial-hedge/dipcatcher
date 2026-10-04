"""Wave-1719 bench adapters: basque-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    basajaun_qa_studies,
    eguzki_qa_studies,
    lamiak_qa_studies,
    mairu_qa_studies,
    mari_qa_studies,
    sugaar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17190


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_basajaun_qa_studies_family(seed: int = _SEED + 0):
    """basajaun_qa_studies: synthetic correctness bench."""
    return _finite_blob(basajaun_qa_studies.bench_basajaun_qa_studies(seed))


def bench_eguzki_qa_studies_family(seed: int = _SEED + 1):
    """eguzki_qa_studies: synthetic correctness bench."""
    return _finite_blob(eguzki_qa_studies.bench_eguzki_qa_studies(seed))


def bench_lamiak_qa_studies_family(seed: int = _SEED + 2):
    """lamiak_qa_studies: synthetic correctness bench."""
    return _finite_blob(lamiak_qa_studies.bench_lamiak_qa_studies(seed))


def bench_mairu_qa_studies_family(seed: int = _SEED + 3):
    """mairu_qa_studies: synthetic correctness bench."""
    return _finite_blob(mairu_qa_studies.bench_mairu_qa_studies(seed))


def bench_mari_qa_studies_family(seed: int = _SEED + 4):
    """mari_qa_studies: synthetic correctness bench."""
    return _finite_blob(mari_qa_studies.bench_mari_qa_studies(seed))


def bench_sugaar_qa_studies_family(seed: int = _SEED + 5):
    """sugaar_qa_studies: synthetic correctness bench."""
    return _finite_blob(sugaar_qa_studies.bench_sugaar_qa_studies(seed))
