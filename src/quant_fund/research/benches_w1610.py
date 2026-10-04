"""Wave-1610 bench adapters: small-mammal-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cavy_qa_studies,
    coypu_qa_studies,
    dhole_qa_studies,
    mara_qa_studies,
    porcupine_qa_studies,
    ratel_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16100


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cavy_qa_studies_family(seed: int = _SEED + 0):
    """cavy_qa_studies: synthetic correctness bench."""
    return _finite_blob(cavy_qa_studies.bench_cavy_qa_studies(seed))


def bench_coypu_qa_studies_family(seed: int = _SEED + 1):
    """coypu_qa_studies: synthetic correctness bench."""
    return _finite_blob(coypu_qa_studies.bench_coypu_qa_studies(seed))


def bench_dhole_qa_studies_family(seed: int = _SEED + 2):
    """dhole_qa_studies: synthetic correctness bench."""
    return _finite_blob(dhole_qa_studies.bench_dhole_qa_studies(seed))


def bench_mara_qa_studies_family(seed: int = _SEED + 3):
    """mara_qa_studies: synthetic correctness bench."""
    return _finite_blob(mara_qa_studies.bench_mara_qa_studies(seed))


def bench_porcupine_qa_studies_family(seed: int = _SEED + 4):
    """porcupine_qa_studies: synthetic correctness bench."""
    return _finite_blob(porcupine_qa_studies.bench_porcupine_qa_studies(seed))


def bench_ratel_qa_studies_family(seed: int = _SEED + 5):
    """ratel_qa_studies: synthetic correctness bench."""
    return _finite_blob(ratel_qa_studies.bench_ratel_qa_studies(seed))
