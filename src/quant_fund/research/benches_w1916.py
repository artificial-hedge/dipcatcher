"""Wave-1916 bench adapters: germanic-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    alp_qa_studies,
    doppelganger_qa_studies,
    kobold_qa_studies,
    mahr_qa_studies,
    poltergeist_qa_studies,
    tatzelwurm_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19160


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alp_qa_studies_family(seed: int = _SEED + 0):
    """alp_qa_studies: synthetic correctness bench."""
    return _finite_blob(alp_qa_studies.bench_alp_qa_studies(seed))


def bench_doppelganger_qa_studies_family(seed: int = _SEED + 1):
    """doppelganger_qa_studies: synthetic correctness bench."""
    return _finite_blob(doppelganger_qa_studies.bench_doppelganger_qa_studies(seed))


def bench_kobold_qa_studies_family(seed: int = _SEED + 2):
    """kobold_qa_studies: synthetic correctness bench."""
    return _finite_blob(kobold_qa_studies.bench_kobold_qa_studies(seed))


def bench_mahr_qa_studies_family(seed: int = _SEED + 3):
    """mahr_qa_studies: synthetic correctness bench."""
    return _finite_blob(mahr_qa_studies.bench_mahr_qa_studies(seed))


def bench_poltergeist_qa_studies_family(seed: int = _SEED + 4):
    """poltergeist_qa_studies: synthetic correctness bench."""
    return _finite_blob(poltergeist_qa_studies.bench_poltergeist_qa_studies(seed))


def bench_tatzelwurm_qa_studies_family(seed: int = _SEED + 5):
    """tatzelwurm_qa_studies: synthetic correctness bench."""
    return _finite_blob(tatzelwurm_qa_studies.bench_tatzelwurm_qa_studies(seed))
