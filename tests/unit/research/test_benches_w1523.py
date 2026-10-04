import pytest

from quant_fund.research import benches_w1523


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bluegrass_qa_studies_family",
        "bench_fescue_qa_studies_family",
        "bench_miscanthus_qa_studies_family",
        "bench_pampas_qa_studies_family",
        "bench_ryegrass_qa_studies_family",
        "bench_switchgrass_qa_studies_family",
    ],
)
def test_benches_w1523(fam):
    out = getattr(benches_w1523, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
