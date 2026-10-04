"""Wave-1287 bench adapters: agent-safety canon (SYNTHETIC only)."""

from quant_fund.models import (
    capability_eval_studies,
    control_eval_studies,
    deception_eval_studies,
    prompt_injection_studies,
    sandbox_escape_studies,
    tool_call_verify_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_capability_eval_studies_family(seed: int = _SEED + 0):
    """capability_eval_studies: synthetic correctness bench."""
    return _finite_blob(capability_eval_studies.bench_capability_eval_studies(seed))


def bench_control_eval_studies_family(seed: int = _SEED + 1):
    """control_eval_studies: synthetic correctness bench."""
    return _finite_blob(control_eval_studies.bench_control_eval_studies(seed))


def bench_deception_eval_studies_family(seed: int = _SEED + 2):
    """deception_eval_studies: synthetic correctness bench."""
    return _finite_blob(deception_eval_studies.bench_deception_eval_studies(seed))


def bench_prompt_injection_studies_family(seed: int = _SEED + 3):
    """prompt_injection_studies: synthetic correctness bench."""
    return _finite_blob(prompt_injection_studies.bench_prompt_injection_studies(seed))


def bench_sandbox_escape_studies_family(seed: int = _SEED + 4):
    """sandbox_escape_studies: synthetic correctness bench."""
    return _finite_blob(sandbox_escape_studies.bench_sandbox_escape_studies(seed))


def bench_tool_call_verify_studies_family(seed: int = _SEED + 5):
    """tool_call_verify_studies: synthetic correctness bench."""
    return _finite_blob(tool_call_verify_studies.bench_tool_call_verify_studies(seed))
