import pytest

from quant_fund.research import benches_w1281


@pytest.mark.parametrize(
    "fam",
    [
        "bench_analogical_prompting_studies_family",
        "bench_graph_of_thought_studies_family",
        "bench_least_to_most_studies_family",
        "bench_plan_and_solve_studies_family",
        "bench_step_back_studies_family",
        "bench_tree_of_thought_studies_family",
    ],
)
def test_benches_w1281(fam):
    out = getattr(benches_w1281, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
