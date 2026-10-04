import pytest

from quant_fund.research import benches_w1674


@pytest.mark.parametrize(
    "fam",
    [
        "bench_genii_qa_studies_family",
        "bench_lares_qa_studies_family",
        "bench_larvae_qa_studies_family",
        "bench_lemures_qa_studies_family",
        "bench_manes_qa_studies_family",
        "bench_penates_qa_studies_family",
    ],
)
def test_benches_w1674(fam):
    out = getattr(benches_w1674, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
