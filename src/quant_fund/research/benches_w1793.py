"""Wave-1793 bench adapters: roman-minor-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cluentia_qa_studies,
    faunus_qa_studies,
    larunda_qa_studies,
    mutina_qa_studies,
    quirinus_qa_studies,
    tellus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17930


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cluentia_qa_studies_family(seed: int = _SEED + 0):
    """cluentia_qa_studies: synthetic correctness bench."""
    return _finite_blob(cluentia_qa_studies.bench_cluentia_qa_studies(seed))


def bench_faunus_qa_studies_family(seed: int = _SEED + 1):
    """faunus_qa_studies: synthetic correctness bench."""
    return _finite_blob(faunus_qa_studies.bench_faunus_qa_studies(seed))


def bench_larunda_qa_studies_family(seed: int = _SEED + 2):
    """larunda_qa_studies: synthetic correctness bench."""
    return _finite_blob(larunda_qa_studies.bench_larunda_qa_studies(seed))


def bench_mutina_qa_studies_family(seed: int = _SEED + 3):
    """mutina_qa_studies: synthetic correctness bench."""
    return _finite_blob(mutina_qa_studies.bench_mutina_qa_studies(seed))


def bench_quirinus_qa_studies_family(seed: int = _SEED + 4):
    """quirinus_qa_studies: synthetic correctness bench."""
    return _finite_blob(quirinus_qa_studies.bench_quirinus_qa_studies(seed))


def bench_tellus_qa_studies_family(seed: int = _SEED + 5):
    """tellus_qa_studies: synthetic correctness bench."""
    return _finite_blob(tellus_qa_studies.bench_tellus_qa_studies(seed))
