"""Wave-1660 bench adapters: greek-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    centaur_qa_studies,
    cyclops_qa_studies,
    griffin_qa_studies,
    hydra_qa_studies,
    medusa_qa_studies,
    sphinx_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_centaur_qa_studies_family(seed: int = _SEED + 0):
    """centaur_qa_studies: synthetic correctness bench."""
    return _finite_blob(centaur_qa_studies.bench_centaur_qa_studies(seed))


def bench_cyclops_qa_studies_family(seed: int = _SEED + 1):
    """cyclops_qa_studies: synthetic correctness bench."""
    return _finite_blob(cyclops_qa_studies.bench_cyclops_qa_studies(seed))


def bench_griffin_qa_studies_family(seed: int = _SEED + 2):
    """griffin_qa_studies: synthetic correctness bench."""
    return _finite_blob(griffin_qa_studies.bench_griffin_qa_studies(seed))


def bench_hydra_qa_studies_family(seed: int = _SEED + 3):
    """hydra_qa_studies: synthetic correctness bench."""
    return _finite_blob(hydra_qa_studies.bench_hydra_qa_studies(seed))


def bench_medusa_qa_studies_family(seed: int = _SEED + 4):
    """medusa_qa_studies: synthetic correctness bench."""
    return _finite_blob(medusa_qa_studies.bench_medusa_qa_studies(seed))


def bench_sphinx_qa_studies_family(seed: int = _SEED + 5):
    """sphinx_qa_studies: synthetic correctness bench."""
    return _finite_blob(sphinx_qa_studies.bench_sphinx_qa_studies(seed))
