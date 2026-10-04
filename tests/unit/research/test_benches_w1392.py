import pytest

from quant_fund.research import benches_w1392


@pytest.mark.parametrize(
    "fam",
    [
        "bench_meta_tool_studies_family",
        "bench_nest_tools_studies_family",
        "bench_toolbench2_studies_family",
        "bench_toolqa_lite_studies_family",
        "bench_ultra_tool_studies_family",
        "bench_work_plus_studies_family",
    ],
)
def test_benches_w1392(fam):
    out = getattr(benches_w1392, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
