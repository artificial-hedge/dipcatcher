import pytest

from quant_fund.research import benches_w1552


@pytest.mark.parametrize(
    "fam",
    [
        "bench_box_turtle_qa_studies_family",
        "bench_map_turtle_qa_studies_family",
        "bench_painted_turtle_qa_studies_family",
        "bench_slider_qa_studies_family",
        "bench_snapping_turtle_qa_studies_family",
        "bench_tortoise_qa_studies_family",
    ],
)
def test_benches_w1552(fam):
    out = getattr(benches_w1552, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
