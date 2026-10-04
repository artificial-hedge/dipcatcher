"""Wave-1558 bench adapters: salmonid canon (SYNTHETIC only)."""

from quant_fund.models import (
    char_qa_studies,
    dolly_varden_qa_studies,
    grayling_qa_studies,
    sockeye_qa_studies,
    steelhead_qa_studies,
    whitefish_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15580


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_char_qa_studies_family(seed: int = _SEED + 0):
    """char_qa_studies: synthetic correctness bench."""
    return _finite_blob(char_qa_studies.bench_char_qa_studies(seed))


def bench_dolly_varden_qa_studies_family(seed: int = _SEED + 1):
    """dolly_varden_qa_studies: synthetic correctness bench."""
    return _finite_blob(dolly_varden_qa_studies.bench_dolly_varden_qa_studies(seed))


def bench_grayling_qa_studies_family(seed: int = _SEED + 2):
    """grayling_qa_studies: synthetic correctness bench."""
    return _finite_blob(grayling_qa_studies.bench_grayling_qa_studies(seed))


def bench_sockeye_qa_studies_family(seed: int = _SEED + 3):
    """sockeye_qa_studies: synthetic correctness bench."""
    return _finite_blob(sockeye_qa_studies.bench_sockeye_qa_studies(seed))


def bench_steelhead_qa_studies_family(seed: int = _SEED + 4):
    """steelhead_qa_studies: synthetic correctness bench."""
    return _finite_blob(steelhead_qa_studies.bench_steelhead_qa_studies(seed))


def bench_whitefish_qa_studies_family(seed: int = _SEED + 5):
    """whitefish_qa_studies: synthetic correctness bench."""
    return _finite_blob(whitefish_qa_studies.bench_whitefish_qa_studies(seed))
