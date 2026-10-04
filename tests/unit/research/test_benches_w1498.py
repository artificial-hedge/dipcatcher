import pytest

from quant_fund.research import benches_w1498


@pytest.mark.parametrize(
    "fam",
    [
        "bench_auklet_qa_studies_family",
        "bench_booby_qa_studies_family",
        "bench_frigatebird_qa_studies_family",
        "bench_guillemot_qa_studies_family",
        "bench_murrelet_qa_studies_family",
        "bench_razorbill_qa_studies_family",
    ],
)
def test_benches_w1498(fam):
    out = getattr(benches_w1498, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
