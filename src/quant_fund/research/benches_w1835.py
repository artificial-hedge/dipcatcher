"""Wave-1835 bench adapters: luwian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    hannahanna2_qa_studies,
    istanuwa2_qa_studies,
    iyarri2_qa_studies,
    kamrusepa2_qa_studies,
    runtija2_qa_studies,
    tarhunza2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18350


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hannahanna2_qa_studies_family(seed: int = _SEED + 0):
    """hannahanna2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hannahanna2_qa_studies.bench_hannahanna2_qa_studies(seed))


def bench_istanuwa2_qa_studies_family(seed: int = _SEED + 1):
    """istanuwa2_qa_studies: synthetic correctness bench."""
    return _finite_blob(istanuwa2_qa_studies.bench_istanuwa2_qa_studies(seed))


def bench_iyarri2_qa_studies_family(seed: int = _SEED + 2):
    """iyarri2_qa_studies: synthetic correctness bench."""
    return _finite_blob(iyarri2_qa_studies.bench_iyarri2_qa_studies(seed))


def bench_kamrusepa2_qa_studies_family(seed: int = _SEED + 3):
    """kamrusepa2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamrusepa2_qa_studies.bench_kamrusepa2_qa_studies(seed))


def bench_runtija2_qa_studies_family(seed: int = _SEED + 4):
    """runtija2_qa_studies: synthetic correctness bench."""
    return _finite_blob(runtija2_qa_studies.bench_runtija2_qa_studies(seed))


def bench_tarhunza2_qa_studies_family(seed: int = _SEED + 5):
    """tarhunza2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarhunza2_qa_studies.bench_tarhunza2_qa_studies(seed))
