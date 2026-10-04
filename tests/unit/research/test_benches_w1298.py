import pytest

from quant_fund.research import benches_w1298


@pytest.mark.parametrize(
    "fam",
    [
        "bench_browse_eval_studies_family",
        "bench_os_world_studies_family",
        "bench_swe_bench_studies_family",
        "bench_terminal_bench_studies_family",
        "bench_tool_use_eval_studies_family",
        "bench_web_arena_studies_family",
    ],
)
def test_benches_w1298(fam):
    out = getattr(benches_w1298, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
