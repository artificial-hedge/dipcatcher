"""Wave-1312 bench adapters: risk-domain canon (SYNTHETIC only)."""

from quant_fund.models import (
    bio_risk_eval_studies,
    chem_risk_eval_studies,
    cyber_sec_eval_studies,
    lab_bench_studies,
    malicious_instruct_studies,
    wmdp_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13120


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bio_risk_eval_studies_family(seed: int = _SEED + 0):
    """bio_risk_eval_studies: synthetic correctness bench."""
    return _finite_blob(bio_risk_eval_studies.bench_bio_risk_eval_studies(seed))


def bench_chem_risk_eval_studies_family(seed: int = _SEED + 1):
    """chem_risk_eval_studies: synthetic correctness bench."""
    return _finite_blob(chem_risk_eval_studies.bench_chem_risk_eval_studies(seed))


def bench_cyber_sec_eval_studies_family(seed: int = _SEED + 2):
    """cyber_sec_eval_studies: synthetic correctness bench."""
    return _finite_blob(cyber_sec_eval_studies.bench_cyber_sec_eval_studies(seed))


def bench_lab_bench_studies_family(seed: int = _SEED + 3):
    """lab_bench_studies: synthetic correctness bench."""
    return _finite_blob(lab_bench_studies.bench_lab_bench_studies(seed))


def bench_malicious_instruct_studies_family(seed: int = _SEED + 4):
    """malicious_instruct_studies: synthetic correctness bench."""
    return _finite_blob(malicious_instruct_studies.bench_malicious_instruct_studies(seed))


def bench_wmdp_studies_family(seed: int = _SEED + 5):
    """wmdp_studies: synthetic correctness bench."""
    return _finite_blob(wmdp_studies.bench_wmdp_studies(seed))
