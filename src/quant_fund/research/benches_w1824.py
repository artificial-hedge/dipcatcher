"""Wave-1824 bench adapters: hungarian-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    boszorka2_qa_studies,
    liderec2_qa_studies,
    remete2_qa_studies,
    sarkany2_qa_studies,
    tatros2_qa_studies,
    turul2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18240


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_boszorka2_qa_studies_family(seed: int = _SEED + 0):
    """boszorka2_qa_studies: synthetic correctness bench."""
    return _finite_blob(boszorka2_qa_studies.bench_boszorka2_qa_studies(seed))


def bench_liderec2_qa_studies_family(seed: int = _SEED + 1):
    """liderec2_qa_studies: synthetic correctness bench."""
    return _finite_blob(liderec2_qa_studies.bench_liderec2_qa_studies(seed))


def bench_remete2_qa_studies_family(seed: int = _SEED + 2):
    """remete2_qa_studies: synthetic correctness bench."""
    return _finite_blob(remete2_qa_studies.bench_remete2_qa_studies(seed))


def bench_sarkany2_qa_studies_family(seed: int = _SEED + 3):
    """sarkany2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sarkany2_qa_studies.bench_sarkany2_qa_studies(seed))


def bench_tatros2_qa_studies_family(seed: int = _SEED + 4):
    """tatros2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tatros2_qa_studies.bench_tatros2_qa_studies(seed))


def bench_turul2_qa_studies_family(seed: int = _SEED + 5):
    """turul2_qa_studies: synthetic correctness bench."""
    return _finite_blob(turul2_qa_studies.bench_turul2_qa_studies(seed))
