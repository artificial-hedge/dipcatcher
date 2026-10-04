import pytest

from quant_fund.research import benches_w1309


@pytest.mark.parametrize(
    "fam",
    [
        "bench_frontier_math_studies_family",
        "bench_gpqa_studies_family",
        "bench_hle_studies_family",
        "bench_mmlu_pro_studies_family",
        "bench_tau_bench_studies_family",
        "bench_workarena_studies_family",
    ],
)
def test_benches_w1309(fam):
    out = getattr(benches_w1309, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
