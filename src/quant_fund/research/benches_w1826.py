"""Wave-1826 bench adapters: turkic-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    alkarisi2_qa_studies,
    baiyz2_qa_studies,
    erlik2_qa_studies,
    kydyr2_qa_studies,
    tenger2_qa_studies,
    ulgen2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18260


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alkarisi2_qa_studies_family(seed: int = _SEED + 0):
    """alkarisi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(alkarisi2_qa_studies.bench_alkarisi2_qa_studies(seed))


def bench_baiyz2_qa_studies_family(seed: int = _SEED + 1):
    """baiyz2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baiyz2_qa_studies.bench_baiyz2_qa_studies(seed))


def bench_erlik2_qa_studies_family(seed: int = _SEED + 2):
    """erlik2_qa_studies: synthetic correctness bench."""
    return _finite_blob(erlik2_qa_studies.bench_erlik2_qa_studies(seed))


def bench_kydyr2_qa_studies_family(seed: int = _SEED + 3):
    """kydyr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kydyr2_qa_studies.bench_kydyr2_qa_studies(seed))


def bench_tenger2_qa_studies_family(seed: int = _SEED + 4):
    """tenger2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tenger2_qa_studies.bench_tenger2_qa_studies(seed))


def bench_ulgen2_qa_studies_family(seed: int = _SEED + 5):
    """ulgen2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ulgen2_qa_studies.bench_ulgen2_qa_studies(seed))
