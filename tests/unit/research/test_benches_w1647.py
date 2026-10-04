import pytest

from quant_fund.research import benches_w1647


@pytest.mark.parametrize(
    "fam",
    [
        "bench_akhekh_qa_studies_family",
        "bench_ammit_qa_studies_family",
        "bench_apophis_qa_studies_family",
        "bench_bes_qa_studies_family",
        "bench_sekhmet_qa_studies_family",
        "bench_sphairo_qa_studies_family",
    ],
)
def test_benches_w1647(fam):
    out = getattr(benches_w1647, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
