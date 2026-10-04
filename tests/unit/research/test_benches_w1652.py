import pytest

from quant_fund.research import benches_w1652


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cuco_qa_studies_family",
        "bench_dahu_qa_studies_family",
        "bench_gargouille_qa_studies_family",
        "bench_lavellan_qa_studies_family",
        "bench_muscaliet_qa_studies_family",
        "bench_tarasque_qa_studies_family",
    ],
)
def test_benches_w1652(fam):
    out = getattr(benches_w1652, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
