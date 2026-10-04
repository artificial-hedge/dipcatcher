import pytest

from quant_fund.research import benches_w1432


@pytest.mark.parametrize(
    "fam",
    [
        "bench_beverage_qa_studies_family",
        "bench_cuisine_qa_studies_family",
        "bench_dessert_qa_studies_family",
        "bench_dish_qa_studies_family",
        "bench_fruit_qa_studies_family",
        "bench_ingredient_qa_studies_family",
    ],
)
def test_benches_w1432(fam):
    out = getattr(benches_w1432, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
