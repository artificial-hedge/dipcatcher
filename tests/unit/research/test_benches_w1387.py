import pytest

from quant_fund.research import benches_w1387


@pytest.mark.parametrize(
    "fam",
    [
        "bench_api_blend_studies_family",
        "bench_bfcl_v3_studies_family",
        "bench_gorilla_eval_studies_family",
        "bench_gta_bench_studies_family",
        "bench_seal_tools_studies_family",
        "bench_stabletoolbench_studies_family",
    ],
)
def test_benches_w1387(fam):
    out = getattr(benches_w1387, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
