"""Wave-1639 bench adapters: yokai-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    dorotabo_qa_studies,
    kitsune_3_qa_studies,
    tanuki_3_qa_studies,
    tengu_2_qa_studies,
    yukionna_qa_studies,
    zashiki_warashi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16390


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dorotabo_qa_studies_family(seed: int = _SEED + 0):
    """dorotabo_qa_studies: synthetic correctness bench."""
    return _finite_blob(dorotabo_qa_studies.bench_dorotabo_qa_studies(seed))


def bench_kitsune_3_qa_studies_family(seed: int = _SEED + 1):
    """kitsune_3_qa_studies: synthetic correctness bench."""
    return _finite_blob(kitsune_3_qa_studies.bench_kitsune_3_qa_studies(seed))


def bench_tanuki_3_qa_studies_family(seed: int = _SEED + 2):
    """tanuki_3_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanuki_3_qa_studies.bench_tanuki_3_qa_studies(seed))


def bench_tengu_2_qa_studies_family(seed: int = _SEED + 3):
    """tengu_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tengu_2_qa_studies.bench_tengu_2_qa_studies(seed))


def bench_yukionna_qa_studies_family(seed: int = _SEED + 4):
    """yukionna_qa_studies: synthetic correctness bench."""
    return _finite_blob(yukionna_qa_studies.bench_yukionna_qa_studies(seed))


def bench_zashiki_warashi_qa_studies_family(seed: int = _SEED + 5):
    """zashiki_warashi_qa_studies: synthetic correctness bench."""
    return _finite_blob(zashiki_warashi_qa_studies.bench_zashiki_warashi_qa_studies(seed))
