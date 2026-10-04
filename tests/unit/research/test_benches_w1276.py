import pytest

from quant_fund.research import benches_w1276


@pytest.mark.parametrize(
    "fam",
    [
        "bench_attribution_patching_studies_family",
        "bench_causal_scrubbing_studies_family",
        "bench_function_vector_studies_family",
        "bench_induction_head_studies_family",
        "bench_monosemantic_studies_family",
        "bench_superposition_studies_family",
    ],
)
def test_benches_w1276(fam):
    out = getattr(benches_w1276, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
