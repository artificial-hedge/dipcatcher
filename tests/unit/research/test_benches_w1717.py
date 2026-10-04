import pytest

from quant_fund.research import benches_w1717


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bindus_qa_studies_family",
        "bench_illyris_qa_studies_family",
        "bench_medaurus_qa_studies_family",
        "bench_redon_qa_studies_family",
        "bench_thana_qa_studies_family",
        "bench_vidasus_qa_studies_family",
    ],
)
def test_benches_w1717(fam):
    out = getattr(benches_w1717, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
