import pytest

from quant_fund.research import benches_w1385


@pytest.mark.parametrize(
    "fam",
    [
        "bench_airtasks_studies_family",
        "bench_browsergym_studies_family",
        "bench_maze_eval_studies_family",
        "bench_mmind2web_studies_family",
        "bench_screenqa_studies_family",
        "bench_weblinx_studies_family",
    ],
)
def test_benches_w1385(fam):
    out = getattr(benches_w1385, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
