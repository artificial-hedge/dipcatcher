import pytest

from quant_fund.research import benches_w1270


@pytest.mark.parametrize(
    "fam",
    [
        "bench_attribution_graph_studies_family",
        "bench_causal_tracing_studies_family",
        "bench_circuit_discovery_studies_family",
        "bench_feature_geometry_studies_family",
        "bench_gated_sae_studies_family",
        "bench_transcoder_studies_family",
    ],
)
def test_benches_w1270(fam):
    out = getattr(benches_w1270, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
