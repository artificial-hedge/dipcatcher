import pytest

from quant_fund.research import benches_w1571


@pytest.mark.parametrize(
    "fam",
    [
        "bench_clam_qa_studies_family",
        "bench_conch_qa_studies_family",
        "bench_mussel_qa_studies_family",
        "bench_oyster_qa_studies_family",
        "bench_scallop_qa_studies_family",
        "bench_whelk_qa_studies_family",
    ],
)
def test_benches_w1571(fam):
    out = getattr(benches_w1571, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
