"""Wave-1707 bench adapters: finno-ugric-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ahti_qa_studies,
    ilmatar_qa_studies,
    kiputytto_qa_studies,
    louhi_qa_studies,
    tapio_qa_studies,
    ukko_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17070


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ahti_qa_studies_family(seed: int = _SEED + 0):
    """ahti_qa_studies: synthetic correctness bench."""
    return _finite_blob(ahti_qa_studies.bench_ahti_qa_studies(seed))


def bench_ilmatar_qa_studies_family(seed: int = _SEED + 1):
    """ilmatar_qa_studies: synthetic correctness bench."""
    return _finite_blob(ilmatar_qa_studies.bench_ilmatar_qa_studies(seed))


def bench_kiputytto_qa_studies_family(seed: int = _SEED + 2):
    """kiputytto_qa_studies: synthetic correctness bench."""
    return _finite_blob(kiputytto_qa_studies.bench_kiputytto_qa_studies(seed))


def bench_louhi_qa_studies_family(seed: int = _SEED + 3):
    """louhi_qa_studies: synthetic correctness bench."""
    return _finite_blob(louhi_qa_studies.bench_louhi_qa_studies(seed))


def bench_tapio_qa_studies_family(seed: int = _SEED + 4):
    """tapio_qa_studies: synthetic correctness bench."""
    return _finite_blob(tapio_qa_studies.bench_tapio_qa_studies(seed))


def bench_ukko_qa_studies_family(seed: int = _SEED + 5):
    """ukko_qa_studies: synthetic correctness bench."""
    return _finite_blob(ukko_qa_studies.bench_ukko_qa_studies(seed))
