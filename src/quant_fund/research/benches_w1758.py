"""Wave-1758 bench adapters: african-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    abiku_qa_studies,
    anansi_qa_studies,
    ifa_qa_studies,
    obatala_qa_studies,
    oya_qa_studies,
    shango_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17580


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abiku_qa_studies_family(seed: int = _SEED + 0):
    """abiku_qa_studies: synthetic correctness bench."""
    return _finite_blob(abiku_qa_studies.bench_abiku_qa_studies(seed))


def bench_anansi_qa_studies_family(seed: int = _SEED + 1):
    """anansi_qa_studies: synthetic correctness bench."""
    return _finite_blob(anansi_qa_studies.bench_anansi_qa_studies(seed))


def bench_ifa_qa_studies_family(seed: int = _SEED + 2):
    """ifa_qa_studies: synthetic correctness bench."""
    return _finite_blob(ifa_qa_studies.bench_ifa_qa_studies(seed))


def bench_obatala_qa_studies_family(seed: int = _SEED + 3):
    """obatala_qa_studies: synthetic correctness bench."""
    return _finite_blob(obatala_qa_studies.bench_obatala_qa_studies(seed))


def bench_oya_qa_studies_family(seed: int = _SEED + 4):
    """oya_qa_studies: synthetic correctness bench."""
    return _finite_blob(oya_qa_studies.bench_oya_qa_studies(seed))


def bench_shango_qa_studies_family(seed: int = _SEED + 5):
    """shango_qa_studies: synthetic correctness bench."""
    return _finite_blob(shango_qa_studies.bench_shango_qa_studies(seed))
