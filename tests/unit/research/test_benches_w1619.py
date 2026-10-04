import pytest

from quant_fund.research import benches_w1619


@pytest.mark.parametrize(
    "fam",
    [
        "bench_blobfish_qa_studies_family",
        "bench_dragonfish_qa_studies_family",
        "bench_dumbo_qa_studies_family",
        "bench_fangtooth_qa_studies_family",
        "bench_gulper_qa_studies_family",
        "bench_tripodfish_qa_studies_family",
    ],
)
def test_benches_w1619(fam):
    out = getattr(benches_w1619, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
