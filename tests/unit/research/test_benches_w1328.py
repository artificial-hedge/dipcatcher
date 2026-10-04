import pytest

from quant_fund.research import benches_w1328


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apps_bench_studies_family",
        "bench_class_eval_studies_family",
        "bench_code_contests_studies_family",
        "bench_multipl_e_studies_family",
        "bench_polyglot_bench_studies_family",
        "bench_repobench_studies_family",
    ],
)
def test_benches_w1328(fam):
    out = getattr(benches_w1328, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
