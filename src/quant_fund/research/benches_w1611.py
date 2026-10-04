"""Wave-1611 bench adapters: lemur-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    crowned_lemur_qa_studies,
    fat_tailed_qa_studies,
    fork_marked_qa_studies,
    needle_clawed_qa_studies,
    ringtail_qa_studies,
    sifaka_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16110


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_crowned_lemur_qa_studies_family(seed: int = _SEED + 0):
    """crowned_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(crowned_lemur_qa_studies.bench_crowned_lemur_qa_studies(seed))


def bench_fat_tailed_qa_studies_family(seed: int = _SEED + 1):
    """fat_tailed_qa_studies: synthetic correctness bench."""
    return _finite_blob(fat_tailed_qa_studies.bench_fat_tailed_qa_studies(seed))


def bench_fork_marked_qa_studies_family(seed: int = _SEED + 2):
    """fork_marked_qa_studies: synthetic correctness bench."""
    return _finite_blob(fork_marked_qa_studies.bench_fork_marked_qa_studies(seed))


def bench_needle_clawed_qa_studies_family(seed: int = _SEED + 3):
    """needle_clawed_qa_studies: synthetic correctness bench."""
    return _finite_blob(needle_clawed_qa_studies.bench_needle_clawed_qa_studies(seed))


def bench_ringtail_qa_studies_family(seed: int = _SEED + 4):
    """ringtail_qa_studies: synthetic correctness bench."""
    return _finite_blob(ringtail_qa_studies.bench_ringtail_qa_studies(seed))


def bench_sifaka_qa_studies_family(seed: int = _SEED + 5):
    """sifaka_qa_studies: synthetic correctness bench."""
    return _finite_blob(sifaka_qa_studies.bench_sifaka_qa_studies(seed))
