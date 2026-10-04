import pytest

from quant_fund.research import benches_w1500


@pytest.mark.parametrize(
    "fam",
    [
        "bench_caddisfly_qa_studies_family",
        "bench_centipede_qa_studies_family",
        "bench_horntail_qa_studies_family",
        "bench_lacewing_qa_studies_family",
        "bench_millipede_qa_studies_family",
        "bench_spider_qa_studies_family",
    ],
)
def test_benches_w1500(fam):
    out = getattr(benches_w1500, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
