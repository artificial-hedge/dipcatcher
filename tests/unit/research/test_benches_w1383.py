import pytest

from quant_fund.research import benches_w1383


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aime24_studies_family",
        "bench_gpqa_diamond_studies_family",
        "bench_hle_lite_studies_family",
        "bench_mmmlu_lite_studies_family",
        "bench_olympiadbench_studies_family",
        "bench_super_gpqa_studies_family",
    ],
)
def test_benches_w1383(fam):
    out = getattr(benches_w1383, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
