"""Wave-1959 bench adapters: goetic-throne canon (SYNTHETIC only)."""

from quant_fund.models import (
    adramelech_qa_studies,
    azazel_qa_studies,
    belphegor_qa_studies,
    ipos_qa_studies,
    marchosias_qa_studies,
    phenex_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adramelech_qa_studies_family(seed: int = _SEED + 0):
    """adramelech_qa_studies: synthetic correctness bench."""
    return _finite_blob(adramelech_qa_studies.bench_adramelech_qa_studies(seed))


def bench_azazel_qa_studies_family(seed: int = _SEED + 1):
    """azazel_qa_studies: synthetic correctness bench."""
    return _finite_blob(azazel_qa_studies.bench_azazel_qa_studies(seed))


def bench_belphegor_qa_studies_family(seed: int = _SEED + 2):
    """belphegor_qa_studies: synthetic correctness bench."""
    return _finite_blob(belphegor_qa_studies.bench_belphegor_qa_studies(seed))


def bench_ipos_qa_studies_family(seed: int = _SEED + 3):
    """ipos_qa_studies: synthetic correctness bench."""
    return _finite_blob(ipos_qa_studies.bench_ipos_qa_studies(seed))


def bench_marchosias_qa_studies_family(seed: int = _SEED + 4):
    """marchosias_qa_studies: synthetic correctness bench."""
    return _finite_blob(marchosias_qa_studies.bench_marchosias_qa_studies(seed))


def bench_phenex_qa_studies_family(seed: int = _SEED + 5):
    """phenex_qa_studies: synthetic correctness bench."""
    return _finite_blob(phenex_qa_studies.bench_phenex_qa_studies(seed))
