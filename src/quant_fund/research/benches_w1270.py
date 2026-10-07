"""Wave-1270 bench adapters: mech-interp-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    attribution_graph_studies,
    causal_tracing_studies,
    circuit_discovery_studies,
    feature_geometry_studies,
    gated_sae_studies,
    transcoder_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12700


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


def bench_attribution_graph_studies_family(seed: int = _SEED + 0):
    """attribution_graph_studies: synthetic correctness bench."""
    return _finite_blob(attribution_graph_studies.bench_attribution_graph_studies(seed))


def bench_causal_tracing_studies_family(seed: int = _SEED + 1):
    """causal_tracing_studies: synthetic correctness bench."""
    return _finite_blob(causal_tracing_studies.bench_causal_tracing_studies(seed))


def bench_circuit_discovery_studies_family(seed: int = _SEED + 2):
    """circuit_discovery_studies: synthetic correctness bench."""
    return _finite_blob(circuit_discovery_studies.bench_circuit_discovery_studies(seed))


def bench_feature_geometry_studies_family(seed: int = _SEED + 3):
    """feature_geometry_studies: synthetic correctness bench."""
    return _finite_blob(feature_geometry_studies.bench_feature_geometry_studies(seed))


def bench_gated_sae_studies_family(seed: int = _SEED + 4):
    """gated_sae_studies: synthetic correctness bench."""
    return _finite_blob(gated_sae_studies.bench_gated_sae_studies(seed))


def bench_transcoder_studies_family(seed: int = _SEED + 5):
    """transcoder_studies: synthetic correctness bench."""
    return _finite_blob(transcoder_studies.bench_transcoder_studies(seed))
