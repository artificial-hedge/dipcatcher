import pytest

from quant_fund.research import benches_w1491


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cypress_qa_studies_family",
        "bench_eucalyptus_qa_studies_family",
        "bench_hemlock_qa_studies_family",
        "bench_laurel_qa_studies_family",
        "bench_magnolia_qa_studies_family",
        "bench_spruce_qa_studies_family",
    ],
)
def test_benches_w1491(fam):
    out = getattr(benches_w1491, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
