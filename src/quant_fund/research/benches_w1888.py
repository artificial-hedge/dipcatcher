"""Wave-1888 bench adapters: eddic-lore canon (SYNTHETIC only)."""

from quant_fund.models import (
    grimnismal_qa_studies,
    havamal_qa_studies,
    lokasenna_qa_studies,
    skirnismal_qa_studies,
    vafthrudnir_qa_studies,
    voluspa_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18880


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_grimnismal_qa_studies_family(seed: int = _SEED + 0):
    """grimnismal_qa_studies: synthetic correctness bench."""
    return _finite_blob(grimnismal_qa_studies.bench_grimnismal_qa_studies(seed))


def bench_havamal_qa_studies_family(seed: int = _SEED + 1):
    """havamal_qa_studies: synthetic correctness bench."""
    return _finite_blob(havamal_qa_studies.bench_havamal_qa_studies(seed))


def bench_lokasenna_qa_studies_family(seed: int = _SEED + 2):
    """lokasenna_qa_studies: synthetic correctness bench."""
    return _finite_blob(lokasenna_qa_studies.bench_lokasenna_qa_studies(seed))


def bench_skirnismal_qa_studies_family(seed: int = _SEED + 3):
    """skirnismal_qa_studies: synthetic correctness bench."""
    return _finite_blob(skirnismal_qa_studies.bench_skirnismal_qa_studies(seed))


def bench_vafthrudnir_qa_studies_family(seed: int = _SEED + 4):
    """vafthrudnir_qa_studies: synthetic correctness bench."""
    return _finite_blob(vafthrudnir_qa_studies.bench_vafthrudnir_qa_studies(seed))


def bench_voluspa_qa_studies_family(seed: int = _SEED + 5):
    """voluspa_qa_studies: synthetic correctness bench."""
    return _finite_blob(voluspa_qa_studies.bench_voluspa_qa_studies(seed))
