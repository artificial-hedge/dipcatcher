"""Wave-1947 bench adapters: vodou-loa canon (SYNTHETIC only)."""

from quant_fund.models import (
    baron_samedi_qa_studies,
    damballa_qa_studies,
    ezili_dantor_qa_studies,
    ogou_feray_qa_studies,
    papa_legba_qa_studies,
    simbi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19470


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baron_samedi_qa_studies_family(seed: int = _SEED + 0):
    """baron_samedi_qa_studies: synthetic correctness bench."""
    return _finite_blob(baron_samedi_qa_studies.bench_baron_samedi_qa_studies(seed))


def bench_damballa_qa_studies_family(seed: int = _SEED + 1):
    """damballa_qa_studies: synthetic correctness bench."""
    return _finite_blob(damballa_qa_studies.bench_damballa_qa_studies(seed))


def bench_ezili_dantor_qa_studies_family(seed: int = _SEED + 2):
    """ezili_dantor_qa_studies: synthetic correctness bench."""
    return _finite_blob(ezili_dantor_qa_studies.bench_ezili_dantor_qa_studies(seed))


def bench_ogou_feray_qa_studies_family(seed: int = _SEED + 3):
    """ogou_feray_qa_studies: synthetic correctness bench."""
    return _finite_blob(ogou_feray_qa_studies.bench_ogou_feray_qa_studies(seed))


def bench_papa_legba_qa_studies_family(seed: int = _SEED + 4):
    """papa_legba_qa_studies: synthetic correctness bench."""
    return _finite_blob(papa_legba_qa_studies.bench_papa_legba_qa_studies(seed))


def bench_simbi_qa_studies_family(seed: int = _SEED + 5):
    """simbi_qa_studies: synthetic correctness bench."""
    return _finite_blob(simbi_qa_studies.bench_simbi_qa_studies(seed))
