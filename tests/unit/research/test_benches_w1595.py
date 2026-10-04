import pytest

from quant_fund.research import benches_w1595


@pytest.mark.parametrize(
    "fam",
    [
        "bench_andean_cat_qa_studies_family",
        "bench_bay_cat_qa_studies_family",
        "bench_flat_headed_qa_studies_family",
        "bench_geoffroys_qa_studies_family",
        "bench_marbled_cat_qa_studies_family",
        "bench_pampas_cat_qa_studies_family",
    ],
)
def test_benches_w1595(fam):
    out = getattr(benches_w1595, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
