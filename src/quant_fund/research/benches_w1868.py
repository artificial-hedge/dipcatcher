"""Wave-1868 bench adapters: carthaginian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anat_punic_qa_studies,
    carthage_punic_qa_studies,
    el_punic_qa_studies,
    hadad_punic_qa_studies,
    moloch_punic_qa_studies,
    reshef_punic_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18680


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anat_punic_qa_studies_family(seed: int = _SEED + 0):
    """anat_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(anat_punic_qa_studies.bench_anat_punic_qa_studies(seed))


def bench_carthage_punic_qa_studies_family(seed: int = _SEED + 1):
    """carthage_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(carthage_punic_qa_studies.bench_carthage_punic_qa_studies(seed))


def bench_el_punic_qa_studies_family(seed: int = _SEED + 2):
    """el_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(el_punic_qa_studies.bench_el_punic_qa_studies(seed))


def bench_hadad_punic_qa_studies_family(seed: int = _SEED + 3):
    """hadad_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(hadad_punic_qa_studies.bench_hadad_punic_qa_studies(seed))


def bench_moloch_punic_qa_studies_family(seed: int = _SEED + 4):
    """moloch_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(moloch_punic_qa_studies.bench_moloch_punic_qa_studies(seed))


def bench_reshef_punic_qa_studies_family(seed: int = _SEED + 5):
    """reshef_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(reshef_punic_qa_studies.bench_reshef_punic_qa_studies(seed))
