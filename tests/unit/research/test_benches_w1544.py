import pytest

from quant_fund.research import benches_w1544


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hoopoe_qa_studies_family",
        "bench_nunbird_qa_studies_family",
        "bench_nunlet_qa_studies_family",
        "bench_puffbird_qa_studies_family",
        "bench_toco_qa_studies_family",
        "bench_woodhoopoe_qa_studies_family",
    ],
)
def test_benches_w1544(fam):
    out = getattr(benches_w1544, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
