"""Wave-1830 bench adapters: hittite-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    apali2_qa_studies,
    arinniti2_qa_studies,
    kumarbi3_qa_studies,
    siyum2_qa_studies,
    wulukanni2_qa_studies,
    zintuhi2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18300


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apali2_qa_studies_family(seed: int = _SEED + 0):
    """apali2_qa_studies: synthetic correctness bench."""
    return _finite_blob(apali2_qa_studies.bench_apali2_qa_studies(seed))


def bench_arinniti2_qa_studies_family(seed: int = _SEED + 1):
    """arinniti2_qa_studies: synthetic correctness bench."""
    return _finite_blob(arinniti2_qa_studies.bench_arinniti2_qa_studies(seed))


def bench_kumarbi3_qa_studies_family(seed: int = _SEED + 2):
    """kumarbi3_qa_studies: synthetic correctness bench."""
    return _finite_blob(kumarbi3_qa_studies.bench_kumarbi3_qa_studies(seed))


def bench_siyum2_qa_studies_family(seed: int = _SEED + 3):
    """siyum2_qa_studies: synthetic correctness bench."""
    return _finite_blob(siyum2_qa_studies.bench_siyum2_qa_studies(seed))


def bench_wulukanni2_qa_studies_family(seed: int = _SEED + 4):
    """wulukanni2_qa_studies: synthetic correctness bench."""
    return _finite_blob(wulukanni2_qa_studies.bench_wulukanni2_qa_studies(seed))


def bench_zintuhi2_qa_studies_family(seed: int = _SEED + 5):
    """zintuhi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(zintuhi2_qa_studies.bench_zintuhi2_qa_studies(seed))
