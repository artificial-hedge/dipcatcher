"""Wave-1876 bench adapters: punic-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    abdir_qa_studies,
    baal_magon_qa_studies,
    melkob_qa_studies,
    safun_hu_qa_studies,
    shadash_qa_studies,
    sinn_bedri_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18760


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abdir_qa_studies_family(seed: int = _SEED + 0):
    """abdir_qa_studies: synthetic correctness bench."""
    return _finite_blob(abdir_qa_studies.bench_abdir_qa_studies(seed))


def bench_baal_magon_qa_studies_family(seed: int = _SEED + 1):
    """baal_magon_qa_studies: synthetic correctness bench."""
    return _finite_blob(baal_magon_qa_studies.bench_baal_magon_qa_studies(seed))


def bench_melkob_qa_studies_family(seed: int = _SEED + 2):
    """melkob_qa_studies: synthetic correctness bench."""
    return _finite_blob(melkob_qa_studies.bench_melkob_qa_studies(seed))


def bench_safun_hu_qa_studies_family(seed: int = _SEED + 3):
    """safun_hu_qa_studies: synthetic correctness bench."""
    return _finite_blob(safun_hu_qa_studies.bench_safun_hu_qa_studies(seed))


def bench_shadash_qa_studies_family(seed: int = _SEED + 4):
    """shadash_qa_studies: synthetic correctness bench."""
    return _finite_blob(shadash_qa_studies.bench_shadash_qa_studies(seed))


def bench_sinn_bedri_qa_studies_family(seed: int = _SEED + 5):
    """sinn_bedri_qa_studies: synthetic correctness bench."""
    return _finite_blob(sinn_bedri_qa_studies.bench_sinn_bedri_qa_studies(seed))
