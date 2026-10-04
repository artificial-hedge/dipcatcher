"""Wave-1566 bench adapters: freshwater-fish canon (SYNTHETIC only)."""

from quant_fund.models import (
    bluegill_qa_studies,
    crappie_qa_studies,
    perch_qa_studies,
    pike_qa_studies,
    sturgeon_qa_studies,
    walleye_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15660


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bluegill_qa_studies_family(seed: int = _SEED + 0):
    """bluegill_qa_studies: synthetic correctness bench."""
    return _finite_blob(bluegill_qa_studies.bench_bluegill_qa_studies(seed))


def bench_crappie_qa_studies_family(seed: int = _SEED + 1):
    """crappie_qa_studies: synthetic correctness bench."""
    return _finite_blob(crappie_qa_studies.bench_crappie_qa_studies(seed))


def bench_perch_qa_studies_family(seed: int = _SEED + 2):
    """perch_qa_studies: synthetic correctness bench."""
    return _finite_blob(perch_qa_studies.bench_perch_qa_studies(seed))


def bench_pike_qa_studies_family(seed: int = _SEED + 3):
    """pike_qa_studies: synthetic correctness bench."""
    return _finite_blob(pike_qa_studies.bench_pike_qa_studies(seed))


def bench_sturgeon_qa_studies_family(seed: int = _SEED + 4):
    """sturgeon_qa_studies: synthetic correctness bench."""
    return _finite_blob(sturgeon_qa_studies.bench_sturgeon_qa_studies(seed))


def bench_walleye_qa_studies_family(seed: int = _SEED + 5):
    """walleye_qa_studies: synthetic correctness bench."""
    return _finite_blob(walleye_qa_studies.bench_walleye_qa_studies(seed))
