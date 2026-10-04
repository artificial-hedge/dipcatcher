"""Wave-1891 bench adapters: folk-spirit lore canon (SYNTHETIC only)."""

from quant_fund.models import (
    bergelmir_qa_studies,
    jormunrek_qa_studies,
    khan_tengri_qa_studies,
    peri_qa_studies,
    umm_sibyan_qa_studies,
    ymir_hrimthurs_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18910


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bergelmir_qa_studies_family(seed: int = _SEED + 0):
    """bergelmir_qa_studies: synthetic correctness bench."""
    return _finite_blob(bergelmir_qa_studies.bench_bergelmir_qa_studies(seed))


def bench_jormunrek_qa_studies_family(seed: int = _SEED + 1):
    """jormunrek_qa_studies: synthetic correctness bench."""
    return _finite_blob(jormunrek_qa_studies.bench_jormunrek_qa_studies(seed))


def bench_khan_tengri_qa_studies_family(seed: int = _SEED + 2):
    """khan_tengri_qa_studies: synthetic correctness bench."""
    return _finite_blob(khan_tengri_qa_studies.bench_khan_tengri_qa_studies(seed))


def bench_peri_qa_studies_family(seed: int = _SEED + 3):
    """peri_qa_studies: synthetic correctness bench."""
    return _finite_blob(peri_qa_studies.bench_peri_qa_studies(seed))


def bench_umm_sibyan_qa_studies_family(seed: int = _SEED + 4):
    """umm_sibyan_qa_studies: synthetic correctness bench."""
    return _finite_blob(umm_sibyan_qa_studies.bench_umm_sibyan_qa_studies(seed))


def bench_ymir_hrimthurs_qa_studies_family(seed: int = _SEED + 5):
    """ymir_hrimthurs_qa_studies: synthetic correctness bench."""
    return _finite_blob(ymir_hrimthurs_qa_studies.bench_ymir_hrimthurs_qa_studies(seed))
