import pytest

from quant_fund.research import benches_w1322


@pytest.mark.parametrize(
    "fam",
    [
        "bench_logic_bench_studies_family",
        "bench_minif2f_studies_family",
        "bench_olympiad_bench_studies_family",
        "bench_putnam_studies_family",
        "bench_truthfulqa_studies_family",
        "bench_zebra_logic_studies_family",
    ],
)
def test_benches_w1322(fam):
    out = getattr(benches_w1322, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
