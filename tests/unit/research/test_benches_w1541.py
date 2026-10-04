import pytest

from quant_fund.research import benches_w1541


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anhinga_qa_studies_family",
        "bench_darter_qa_studies_family",
        "bench_diving_petrel_qa_studies_family",
        "bench_gadfly_qa_studies_family",
        "bench_manx_qa_studies_family",
        "bench_mollymawk_qa_studies_family",
    ],
)
def test_benches_w1541(fam):
    out = getattr(benches_w1541, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
