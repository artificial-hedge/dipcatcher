"""Wave-1896 bench adapters: mesopotamian-demon-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ereshkigal_namtar_qa_studies,
    lama_demon_qa_studies,
    mukil_qa_studies,
    mukil_res_lemuttim_qa_studies,
    nergal_demon_qa_studies,
    rabisu_hursag_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18960


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ereshkigal_namtar_qa_studies_family(seed: int = _SEED + 0):
    """ereshkigal_namtar_qa_studies: synthetic correctness bench."""
    return _finite_blob(ereshkigal_namtar_qa_studies.bench_ereshkigal_namtar_qa_studies(seed))


def bench_lama_demon_qa_studies_family(seed: int = _SEED + 1):
    """lama_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(lama_demon_qa_studies.bench_lama_demon_qa_studies(seed))


def bench_mukil_qa_studies_family(seed: int = _SEED + 2):
    """mukil_qa_studies: synthetic correctness bench."""
    return _finite_blob(mukil_qa_studies.bench_mukil_qa_studies(seed))


def bench_mukil_res_lemuttim_qa_studies_family(seed: int = _SEED + 3):
    """mukil_res_lemuttim_qa_studies: synthetic correctness bench."""
    return _finite_blob(mukil_res_lemuttim_qa_studies.bench_mukil_res_lemuttim_qa_studies(seed))


def bench_nergal_demon_qa_studies_family(seed: int = _SEED + 4):
    """nergal_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(nergal_demon_qa_studies.bench_nergal_demon_qa_studies(seed))


def bench_rabisu_hursag_qa_studies_family(seed: int = _SEED + 5):
    """rabisu_hursag_qa_studies: synthetic correctness bench."""
    return _finite_blob(rabisu_hursag_qa_studies.bench_rabisu_hursag_qa_studies(seed))
