"""Wave-1631 bench adapters: legendary-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    basilisk_qa_studies,
    chimera_qa_studies,
    gorgon_qa_studies,
    griffin_2_qa_studies,
    hydra_2_qa_studies,
    manticore_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16310


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_basilisk_qa_studies_family(seed: int = _SEED + 0):
    """basilisk_qa_studies: synthetic correctness bench."""
    return _finite_blob(basilisk_qa_studies.bench_basilisk_qa_studies(seed))


def bench_chimera_qa_studies_family(seed: int = _SEED + 1):
    """chimera_qa_studies: synthetic correctness bench."""
    return _finite_blob(chimera_qa_studies.bench_chimera_qa_studies(seed))


def bench_gorgon_qa_studies_family(seed: int = _SEED + 2):
    """gorgon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gorgon_qa_studies.bench_gorgon_qa_studies(seed))


def bench_griffin_2_qa_studies_family(seed: int = _SEED + 3):
    """griffin_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(griffin_2_qa_studies.bench_griffin_2_qa_studies(seed))


def bench_hydra_2_qa_studies_family(seed: int = _SEED + 4):
    """hydra_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hydra_2_qa_studies.bench_hydra_2_qa_studies(seed))


def bench_manticore_qa_studies_family(seed: int = _SEED + 5):
    """manticore_qa_studies: synthetic correctness bench."""
    return _finite_blob(manticore_qa_studies.bench_manticore_qa_studies(seed))
