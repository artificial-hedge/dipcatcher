"""Wave-1694 bench adapters: filipino-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    duwende_qa_studies,
    karibusa_qa_studies,
    mambabarang_qa_studies,
    mangkukulam_qa_studies,
    sokoy_qa_studies,
    tiktik_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_duwende_qa_studies_family(seed: int = _SEED + 0):
    """duwende_qa_studies: synthetic correctness bench."""
    return _finite_blob(duwende_qa_studies.bench_duwende_qa_studies(seed))


def bench_karibusa_qa_studies_family(seed: int = _SEED + 1):
    """karibusa_qa_studies: synthetic correctness bench."""
    return _finite_blob(karibusa_qa_studies.bench_karibusa_qa_studies(seed))


def bench_mambabarang_qa_studies_family(seed: int = _SEED + 2):
    """mambabarang_qa_studies: synthetic correctness bench."""
    return _finite_blob(mambabarang_qa_studies.bench_mambabarang_qa_studies(seed))


def bench_mangkukulam_qa_studies_family(seed: int = _SEED + 3):
    """mangkukulam_qa_studies: synthetic correctness bench."""
    return _finite_blob(mangkukulam_qa_studies.bench_mangkukulam_qa_studies(seed))


def bench_sokoy_qa_studies_family(seed: int = _SEED + 4):
    """sokoy_qa_studies: synthetic correctness bench."""
    return _finite_blob(sokoy_qa_studies.bench_sokoy_qa_studies(seed))


def bench_tiktik_qa_studies_family(seed: int = _SEED + 5):
    """tiktik_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiktik_qa_studies.bench_tiktik_qa_studies(seed))
