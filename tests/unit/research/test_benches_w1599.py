import pytest

from quant_fund.research import benches_w1599


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aye_aye_qa_studies_family",
        "bench_howler_qa_studies_family",
        "bench_mouse_lemur_qa_studies_family",
        "bench_night_monkey_qa_studies_family",
        "bench_ring_tailed_qa_studies_family",
        "bench_spider_monkey_qa_studies_family",
    ],
)
def test_benches_w1599(fam):
    out = getattr(benches_w1599, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
