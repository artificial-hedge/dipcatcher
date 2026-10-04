import pytest

from quant_fund.research import benches_w1572


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aster_qa_studies_family",
        "bench_bluebell_qa_studies_family",
        "bench_buttercup_qa_studies_family",
        "bench_columbine_qa_studies_family",
        "bench_cornflower_qa_studies_family",
        "bench_lupine_qa_studies_family",
    ],
)
def test_benches_w1572(fam):
    out = getattr(benches_w1572, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
