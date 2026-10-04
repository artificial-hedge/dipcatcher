"""Wave-1815 bench adapters: greek-myth-11 canon (SYNTHETIC only)."""

from quant_fund.models import (
    demeter2_qa_studies,
    hecate2_qa_studies,
    hestia2_qa_studies,
    iris2_qa_studies,
    nike2_qa_studies,
    persephone2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18150


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_demeter2_qa_studies_family(seed: int = _SEED + 0):
    """demeter2_qa_studies: synthetic correctness bench."""
    return _finite_blob(demeter2_qa_studies.bench_demeter2_qa_studies(seed))


def bench_hecate2_qa_studies_family(seed: int = _SEED + 1):
    """hecate2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hecate2_qa_studies.bench_hecate2_qa_studies(seed))


def bench_hestia2_qa_studies_family(seed: int = _SEED + 2):
    """hestia2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hestia2_qa_studies.bench_hestia2_qa_studies(seed))


def bench_iris2_qa_studies_family(seed: int = _SEED + 3):
    """iris2_qa_studies: synthetic correctness bench."""
    return _finite_blob(iris2_qa_studies.bench_iris2_qa_studies(seed))


def bench_nike2_qa_studies_family(seed: int = _SEED + 4):
    """nike2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nike2_qa_studies.bench_nike2_qa_studies(seed))


def bench_persephone2_qa_studies_family(seed: int = _SEED + 5):
    """persephone2_qa_studies: synthetic correctness bench."""
    return _finite_blob(persephone2_qa_studies.bench_persephone2_qa_studies(seed))
