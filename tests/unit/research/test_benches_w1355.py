import pytest

from quant_fund.research import benches_w1355


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arc_hard2_studies_family",
        "bench_csqa_lite_studies_family",
        "bench_hellaswag_lite_studies_family",
        "bench_piqa_lite_studies_family",
        "bench_prost_lite_studies_family",
        "bench_swag_lite_studies_family",
    ],
)
def test_benches_w1355(fam):
    out = getattr(benches_w1355, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
