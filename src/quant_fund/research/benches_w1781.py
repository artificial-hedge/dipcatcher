"""Wave-1781 bench adapters: greek-myth-8 canon (SYNTHETIC only)."""

from quant_fund.models import (
    apollo_qa_studies,
    artemis_qa_studies,
    athena_qa_studies,
    demeter_qa_studies,
    hera_qa_studies,
    persephone_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17810


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apollo_qa_studies_family(seed: int = _SEED + 0):
    """apollo_qa_studies: synthetic correctness bench."""
    return _finite_blob(apollo_qa_studies.bench_apollo_qa_studies(seed))


def bench_artemis_qa_studies_family(seed: int = _SEED + 1):
    """artemis_qa_studies: synthetic correctness bench."""
    return _finite_blob(artemis_qa_studies.bench_artemis_qa_studies(seed))


def bench_athena_qa_studies_family(seed: int = _SEED + 2):
    """athena_qa_studies: synthetic correctness bench."""
    return _finite_blob(athena_qa_studies.bench_athena_qa_studies(seed))


def bench_demeter_qa_studies_family(seed: int = _SEED + 3):
    """demeter_qa_studies: synthetic correctness bench."""
    return _finite_blob(demeter_qa_studies.bench_demeter_qa_studies(seed))


def bench_hera_qa_studies_family(seed: int = _SEED + 4):
    """hera_qa_studies: synthetic correctness bench."""
    return _finite_blob(hera_qa_studies.bench_hera_qa_studies(seed))


def bench_persephone_qa_studies_family(seed: int = _SEED + 5):
    """persephone_qa_studies: synthetic correctness bench."""
    return _finite_blob(persephone_qa_studies.bench_persephone_qa_studies(seed))
