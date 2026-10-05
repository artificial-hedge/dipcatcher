import pytest

from quant_fund.research import benches_w1815


@pytest.mark.parametrize(
    "fam",
    [
        "bench_demeter2_qa_studies_family",
        "bench_hecate2_qa_studies_family",
        "bench_hestia2_qa_studies_family",
        "bench_iris2_qa_studies_family",
        "bench_nike2_qa_studies_family",
        "bench_persephone2_qa_studies_family",
    ],
)
def test_benches_w1815(fam):
    out = getattr(benches_w1815, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
