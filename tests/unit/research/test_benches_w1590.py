import pytest

from quant_fund.research import benches_w1590


@pytest.mark.parametrize(
    "fam",
    [
        "bench_black_footed_qa_studies_family",
        "bench_fishing_cat_qa_studies_family",
        "bench_jungle_cat_qa_studies_family",
        "bench_pallas_qa_studies_family",
        "bench_rusty_spotted_qa_studies_family",
        "bench_sand_cat_qa_studies_family",
    ],
)
def test_benches_w1590(fam):
    out = getattr(benches_w1590, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
