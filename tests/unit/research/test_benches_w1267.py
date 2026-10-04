import pytest

from quant_fund.research import benches_w1267


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alpha_tensor_studies_family",
        "bench_differentiable_sat_studies_family",
        "bench_neural_theorem_studies_family",
        "bench_program_synthesis_studies_family",
        "bench_sketch_programming_studies_family",
        "bench_symbolic_regression_dl_studies_family",
    ],
)
def test_benches_w1267(fam):
    out = getattr(benches_w1267, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
