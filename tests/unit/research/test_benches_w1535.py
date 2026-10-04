import pytest

from quant_fund.research import benches_w1535


@pytest.mark.parametrize(
    "fam",
    [
        "bench_martin_qa_studies_family",
        "bench_needletail_qa_studies_family",
        "bench_swallow_qa_studies_family",
        "bench_swift_qa_studies_family",
        "bench_swiftlet_qa_studies_family",
        "bench_treeswift_qa_studies_family",
    ],
)
def test_benches_w1535(fam):
    out = getattr(benches_w1535, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
