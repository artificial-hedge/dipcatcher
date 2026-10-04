"""Wave-1530 bench adapters: alloy canon (SYNTHETIC only)."""

from quant_fund.models import (
    amalgam_qa_studies,
    brass_qa_studies,
    bronze_qa_studies,
    nichrome_qa_studies,
    pewter_qa_studies,
    solder_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15300


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amalgam_qa_studies_family(seed: int = _SEED + 0):
    """amalgam_qa_studies: synthetic correctness bench."""
    return _finite_blob(amalgam_qa_studies.bench_amalgam_qa_studies(seed))


def bench_brass_qa_studies_family(seed: int = _SEED + 1):
    """brass_qa_studies: synthetic correctness bench."""
    return _finite_blob(brass_qa_studies.bench_brass_qa_studies(seed))


def bench_bronze_qa_studies_family(seed: int = _SEED + 2):
    """bronze_qa_studies: synthetic correctness bench."""
    return _finite_blob(bronze_qa_studies.bench_bronze_qa_studies(seed))


def bench_nichrome_qa_studies_family(seed: int = _SEED + 3):
    """nichrome_qa_studies: synthetic correctness bench."""
    return _finite_blob(nichrome_qa_studies.bench_nichrome_qa_studies(seed))


def bench_pewter_qa_studies_family(seed: int = _SEED + 4):
    """pewter_qa_studies: synthetic correctness bench."""
    return _finite_blob(pewter_qa_studies.bench_pewter_qa_studies(seed))


def bench_solder_qa_studies_family(seed: int = _SEED + 5):
    """solder_qa_studies: synthetic correctness bench."""
    return _finite_blob(solder_qa_studies.bench_solder_qa_studies(seed))
