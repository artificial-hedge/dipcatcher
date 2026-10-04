import pytest

from quant_fund.research import benches_w1493


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bongo_qa_studies_family",
        "bench_duiker_qa_studies_family",
        "bench_hartebeest_qa_studies_family",
        "bench_nyala_qa_studies_family",
        "bench_topi_qa_studies_family",
        "bench_waterbuck_qa_studies_family",
    ],
)
def test_benches_w1493(fam):
    out = getattr(benches_w1493, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
