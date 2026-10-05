"""Wave-1873 bench adapters: breton-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ar_marzh_qa_studies,
    darkman_qa_studies,
    kaier_qa_studies,
    noz_vat_qa_studies,
    paotr_bugel_qa_studies,
    santez_nonna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18730


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ar_marzh_qa_studies_family(seed: int = _SEED + 0):
    """ar_marzh_qa_studies: synthetic correctness bench."""
    return _finite_blob(ar_marzh_qa_studies.bench_ar_marzh_qa_studies(seed))


def bench_darkman_qa_studies_family(seed: int = _SEED + 1):
    """darkman_qa_studies: synthetic correctness bench."""
    return _finite_blob(darkman_qa_studies.bench_darkman_qa_studies(seed))


def bench_kaier_qa_studies_family(seed: int = _SEED + 2):
    """kaier_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaier_qa_studies.bench_kaier_qa_studies(seed))


def bench_noz_vat_qa_studies_family(seed: int = _SEED + 3):
    """noz_vat_qa_studies: synthetic correctness bench."""
    return _finite_blob(noz_vat_qa_studies.bench_noz_vat_qa_studies(seed))


def bench_paotr_bugel_qa_studies_family(seed: int = _SEED + 4):
    """paotr_bugel_qa_studies: synthetic correctness bench."""
    return _finite_blob(paotr_bugel_qa_studies.bench_paotr_bugel_qa_studies(seed))


def bench_santez_nonna_qa_studies_family(seed: int = _SEED + 5):
    """santez_nonna_qa_studies: synthetic correctness bench."""
    return _finite_blob(santez_nonna_qa_studies.bench_santez_nonna_qa_studies(seed))
