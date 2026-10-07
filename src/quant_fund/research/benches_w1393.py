"""Wave-1393 bench adapters: MCP-web canon (SYNTHETIC only)."""

from quant_fund.models import (
    hamming_mcp_studies,
    mcp_bench_studies,
    net_hack_studies,
    tool_sandbox_studies,
    videoweb_studies,
    webshop_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13930


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hamming_mcp_studies_family(seed: int = _SEED + 0):
    """hamming_mcp_studies: synthetic correctness bench."""
    return _finite_blob(hamming_mcp_studies.bench_hamming_mcp_studies(seed))


def bench_mcp_bench_studies_family(seed: int = _SEED + 1):
    """mcp_bench_studies: synthetic correctness bench."""
    return _finite_blob(mcp_bench_studies.bench_mcp_bench_studies(seed))


def bench_net_hack_studies_family(seed: int = _SEED + 2):
    """net_hack_studies: synthetic correctness bench."""
    return _finite_blob(net_hack_studies.bench_net_hack_studies(seed))


def bench_tool_sandbox_studies_family(seed: int = _SEED + 3):
    """tool_sandbox_studies: synthetic correctness bench."""
    return _finite_blob(tool_sandbox_studies.bench_tool_sandbox_studies(seed))


def bench_videoweb_studies_family(seed: int = _SEED + 4):
    """videoweb_studies: synthetic correctness bench."""
    return _finite_blob(videoweb_studies.bench_videoweb_studies(seed))


def bench_webshop_lite_studies_family(seed: int = _SEED + 5):
    """webshop_lite_studies: synthetic correctness bench."""
    return _finite_blob(webshop_lite_studies.bench_webshop_lite_studies(seed))
