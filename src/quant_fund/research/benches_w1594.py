"""Wave-1594 bench adapters: primate-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bonobo_qa_studies,
    chimpanzee_qa_studies,
    douc_qa_studies,
    proboscis_qa_studies,
    siamang_qa_studies,
    snub_nosed_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bonobo_qa_studies_family(seed: int = _SEED + 0):
    """bonobo_qa_studies: synthetic correctness bench."""
    return _finite_blob(bonobo_qa_studies.bench_bonobo_qa_studies(seed))


def bench_chimpanzee_qa_studies_family(seed: int = _SEED + 1):
    """chimpanzee_qa_studies: synthetic correctness bench."""
    return _finite_blob(chimpanzee_qa_studies.bench_chimpanzee_qa_studies(seed))


def bench_douc_qa_studies_family(seed: int = _SEED + 2):
    """douc_qa_studies: synthetic correctness bench."""
    return _finite_blob(douc_qa_studies.bench_douc_qa_studies(seed))


def bench_proboscis_qa_studies_family(seed: int = _SEED + 3):
    """proboscis_qa_studies: synthetic correctness bench."""
    return _finite_blob(proboscis_qa_studies.bench_proboscis_qa_studies(seed))


def bench_siamang_qa_studies_family(seed: int = _SEED + 4):
    """siamang_qa_studies: synthetic correctness bench."""
    return _finite_blob(siamang_qa_studies.bench_siamang_qa_studies(seed))


def bench_snub_nosed_qa_studies_family(seed: int = _SEED + 5):
    """snub_nosed_qa_studies: synthetic correctness bench."""
    return _finite_blob(snub_nosed_qa_studies.bench_snub_nosed_qa_studies(seed))
