import pytest

from quant_fund.research import benches_w1266


@pytest.mark.parametrize(
    "fam",
    [
        "bench_diffusion_lm_studies_family",
        "bench_kv_compression_studies_family",
        "bench_medusa_speculation_studies_family",
        "bench_moe_shared_expert_studies_family",
        "bench_rope_scaling_studies_family",
        "bench_sparse_attention_studies_family",
    ],
)
def test_benches_w1266(fam):
    out = getattr(benches_w1266, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
