"""Wave-1710 bench adapters: canaanite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    anat_qa_studies,
    asherah_qa_studies,
    baal_qa_studies,
    lotan_qa_studies,
    mot_qa_studies,
    yam_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17100


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anat_qa_studies_family(seed: int = _SEED + 0):
    """anat_qa_studies: synthetic correctness bench."""
    return _finite_blob(anat_qa_studies.bench_anat_qa_studies(seed))


def bench_asherah_qa_studies_family(seed: int = _SEED + 1):
    """asherah_qa_studies: synthetic correctness bench."""
    return _finite_blob(asherah_qa_studies.bench_asherah_qa_studies(seed))


def bench_baal_qa_studies_family(seed: int = _SEED + 2):
    """baal_qa_studies: synthetic correctness bench."""
    return _finite_blob(baal_qa_studies.bench_baal_qa_studies(seed))


def bench_lotan_qa_studies_family(seed: int = _SEED + 3):
    """lotan_qa_studies: synthetic correctness bench."""
    return _finite_blob(lotan_qa_studies.bench_lotan_qa_studies(seed))


def bench_mot_qa_studies_family(seed: int = _SEED + 4):
    """mot_qa_studies: synthetic correctness bench."""
    return _finite_blob(mot_qa_studies.bench_mot_qa_studies(seed))


def bench_yam_qa_studies_family(seed: int = _SEED + 5):
    """yam_qa_studies: synthetic correctness bench."""
    return _finite_blob(yam_qa_studies.bench_yam_qa_studies(seed))
