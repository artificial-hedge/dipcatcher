"""Wave-1679 bench adapters: norse-realm-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    alfheim_qa_studies,
    bergrisi_qa_studies,
    geirahod_qa_studies,
    huldra_qa_studies,
    troll_qa_studies,
    vaetter_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16790


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alfheim_qa_studies_family(seed: int = _SEED + 0):
    """alfheim_qa_studies: synthetic correctness bench."""
    return _finite_blob(alfheim_qa_studies.bench_alfheim_qa_studies(seed))


def bench_bergrisi_qa_studies_family(seed: int = _SEED + 1):
    """bergrisi_qa_studies: synthetic correctness bench."""
    return _finite_blob(bergrisi_qa_studies.bench_bergrisi_qa_studies(seed))


def bench_geirahod_qa_studies_family(seed: int = _SEED + 2):
    """geirahod_qa_studies: synthetic correctness bench."""
    return _finite_blob(geirahod_qa_studies.bench_geirahod_qa_studies(seed))


def bench_huldra_qa_studies_family(seed: int = _SEED + 3):
    """huldra_qa_studies: synthetic correctness bench."""
    return _finite_blob(huldra_qa_studies.bench_huldra_qa_studies(seed))


def bench_troll_qa_studies_family(seed: int = _SEED + 4):
    """troll_qa_studies: synthetic correctness bench."""
    return _finite_blob(troll_qa_studies.bench_troll_qa_studies(seed))


def bench_vaetter_qa_studies_family(seed: int = _SEED + 5):
    """vaetter_qa_studies: synthetic correctness bench."""
    return _finite_blob(vaetter_qa_studies.bench_vaetter_qa_studies(seed))
