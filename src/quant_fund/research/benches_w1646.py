"""Wave-1646 bench adapters: norse-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    fenrir_2_qa_studies,
    garm_qa_studies,
    jormungandr_qa_studies,
    nidhogg_qa_studies,
    ratatoskr_qa_studies,
    sleipnir_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16460


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fenrir_2_qa_studies_family(seed: int = _SEED + 0):
    """fenrir_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(fenrir_2_qa_studies.bench_fenrir_2_qa_studies(seed))


def bench_garm_qa_studies_family(seed: int = _SEED + 1):
    """garm_qa_studies: synthetic correctness bench."""
    return _finite_blob(garm_qa_studies.bench_garm_qa_studies(seed))


def bench_jormungandr_qa_studies_family(seed: int = _SEED + 2):
    """jormungandr_qa_studies: synthetic correctness bench."""
    return _finite_blob(jormungandr_qa_studies.bench_jormungandr_qa_studies(seed))


def bench_nidhogg_qa_studies_family(seed: int = _SEED + 3):
    """nidhogg_qa_studies: synthetic correctness bench."""
    return _finite_blob(nidhogg_qa_studies.bench_nidhogg_qa_studies(seed))


def bench_ratatoskr_qa_studies_family(seed: int = _SEED + 4):
    """ratatoskr_qa_studies: synthetic correctness bench."""
    return _finite_blob(ratatoskr_qa_studies.bench_ratatoskr_qa_studies(seed))


def bench_sleipnir_qa_studies_family(seed: int = _SEED + 5):
    """sleipnir_qa_studies: synthetic correctness bench."""
    return _finite_blob(sleipnir_qa_studies.bench_sleipnir_qa_studies(seed))
