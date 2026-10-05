"""Wave-1935 bench adapters: andean-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    anchancho_qa_studies,
    jarjacha_qa_studies,
    kharisiri_qa_studies,
    muki_qa_studies,
    pishtaco_qa_studies,
    sirenito_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19350


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anchancho_qa_studies_family(seed: int = _SEED + 0):
    """anchancho_qa_studies: synthetic correctness bench."""
    return _finite_blob(anchancho_qa_studies.bench_anchancho_qa_studies(seed))



def bench_jarjacha_qa_studies_family(seed: int = _SEED + 1):
    """jarjacha_qa_studies: synthetic correctness bench."""
    return _finite_blob(jarjacha_qa_studies.bench_jarjacha_qa_studies(seed))



def bench_kharisiri_qa_studies_family(seed: int = _SEED + 2):
    """kharisiri_qa_studies: synthetic correctness bench."""
    return _finite_blob(kharisiri_qa_studies.bench_kharisiri_qa_studies(seed))



def bench_muki_qa_studies_family(seed: int = _SEED + 3):
    """muki_qa_studies: synthetic correctness bench."""
    return _finite_blob(muki_qa_studies.bench_muki_qa_studies(seed))



def bench_pishtaco_qa_studies_family(seed: int = _SEED + 4):
    """pishtaco_qa_studies: synthetic correctness bench."""
    return _finite_blob(pishtaco_qa_studies.bench_pishtaco_qa_studies(seed))



def bench_sirenito_qa_studies_family(seed: int = _SEED + 5):
    """sirenito_qa_studies: synthetic correctness bench."""
    return _finite_blob(sirenito_qa_studies.bench_sirenito_qa_studies(seed))
