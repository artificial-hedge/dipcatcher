"""Wave-1848 bench adapters: meroitic-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    apedemak_qa_studies,
    arensnuphis_qa_studies,
    dedwen_qa_studies,
    mandulis_qa_studies,
    miket_qa_studies,
    sebiumeker_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18480


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apedemak_qa_studies_family(seed: int = _SEED + 0):
    """apedemak_qa_studies: synthetic correctness bench."""
    return _finite_blob(apedemak_qa_studies.bench_apedemak_qa_studies(seed))


def bench_arensnuphis_qa_studies_family(seed: int = _SEED + 1):
    """arensnuphis_qa_studies: synthetic correctness bench."""
    return _finite_blob(arensnuphis_qa_studies.bench_arensnuphis_qa_studies(seed))


def bench_dedwen_qa_studies_family(seed: int = _SEED + 2):
    """dedwen_qa_studies: synthetic correctness bench."""
    return _finite_blob(dedwen_qa_studies.bench_dedwen_qa_studies(seed))


def bench_mandulis_qa_studies_family(seed: int = _SEED + 3):
    """mandulis_qa_studies: synthetic correctness bench."""
    return _finite_blob(mandulis_qa_studies.bench_mandulis_qa_studies(seed))


def bench_miket_qa_studies_family(seed: int = _SEED + 4):
    """miket_qa_studies: synthetic correctness bench."""
    return _finite_blob(miket_qa_studies.bench_miket_qa_studies(seed))


def bench_sebiumeker_qa_studies_family(seed: int = _SEED + 5):
    """sebiumeker_qa_studies: synthetic correctness bench."""
    return _finite_blob(sebiumeker_qa_studies.bench_sebiumeker_qa_studies(seed))
