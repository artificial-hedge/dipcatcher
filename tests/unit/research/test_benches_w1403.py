import pytest

from quant_fund.research import benches_w1403


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ai2d_lite_studies_family",
        "bench_chart_qa_lite_studies_family",
        "bench_docvqa_lite_studies_family",
        "bench_infovqa_lite_studies_family",
        "bench_mmqa_lite_studies_family",
        "bench_ocrvqa_lite_studies_family",
    ],
)
def test_benches_w1403(fam):
    out = getattr(benches_w1403, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
