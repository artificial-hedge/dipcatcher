"""Wave-1851 bench adapters: guanche-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    achaman_qa_studies,
    achuguayo_qa_studies,
    chaxiraxi_qa_studies,
    guayota_qa_studies,
    magec_qa_studies,
    tibicena_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18510


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_achaman_qa_studies_family(seed: int = _SEED + 0):
    """achaman_qa_studies: synthetic correctness bench."""
    return _finite_blob(achaman_qa_studies.bench_achaman_qa_studies(seed))


def bench_achuguayo_qa_studies_family(seed: int = _SEED + 1):
    """achuguayo_qa_studies: synthetic correctness bench."""
    return _finite_blob(achuguayo_qa_studies.bench_achuguayo_qa_studies(seed))


def bench_chaxiraxi_qa_studies_family(seed: int = _SEED + 2):
    """chaxiraxi_qa_studies: synthetic correctness bench."""
    return _finite_blob(chaxiraxi_qa_studies.bench_chaxiraxi_qa_studies(seed))


def bench_guayota_qa_studies_family(seed: int = _SEED + 3):
    """guayota_qa_studies: synthetic correctness bench."""
    return _finite_blob(guayota_qa_studies.bench_guayota_qa_studies(seed))


def bench_magec_qa_studies_family(seed: int = _SEED + 4):
    """magec_qa_studies: synthetic correctness bench."""
    return _finite_blob(magec_qa_studies.bench_magec_qa_studies(seed))


def bench_tibicena_qa_studies_family(seed: int = _SEED + 5):
    """tibicena_qa_studies: synthetic correctness bench."""
    return _finite_blob(tibicena_qa_studies.bench_tibicena_qa_studies(seed))
