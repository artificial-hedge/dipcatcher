"""Wave-1588 bench adapters: pinniped-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bearded_seal_qa_studies,
    crabeater_qa_studies,
    hooded_seal_qa_studies,
    ribbon_seal_qa_studies,
    ringed_seal_qa_studies,
    ross_seal_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15880


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bearded_seal_qa_studies_family(seed: int = _SEED + 0):
    """bearded_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(bearded_seal_qa_studies.bench_bearded_seal_qa_studies(seed))


def bench_crabeater_qa_studies_family(seed: int = _SEED + 1):
    """crabeater_qa_studies: synthetic correctness bench."""
    return _finite_blob(crabeater_qa_studies.bench_crabeater_qa_studies(seed))


def bench_hooded_seal_qa_studies_family(seed: int = _SEED + 2):
    """hooded_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(hooded_seal_qa_studies.bench_hooded_seal_qa_studies(seed))


def bench_ribbon_seal_qa_studies_family(seed: int = _SEED + 3):
    """ribbon_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(ribbon_seal_qa_studies.bench_ribbon_seal_qa_studies(seed))


def bench_ringed_seal_qa_studies_family(seed: int = _SEED + 4):
    """ringed_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(ringed_seal_qa_studies.bench_ringed_seal_qa_studies(seed))


def bench_ross_seal_qa_studies_family(seed: int = _SEED + 5):
    """ross_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(ross_seal_qa_studies.bench_ross_seal_qa_studies(seed))
