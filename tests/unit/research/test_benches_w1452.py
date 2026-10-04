import pytest

from quant_fund.research import benches_w1452


@pytest.mark.parametrize(
    "fam",
    [
        "bench_carrot_qa_studies_family",
        "bench_cucumber_qa_studies_family",
        "bench_garlic_qa_studies_family",
        "bench_onion_qa_studies_family",
        "bench_potato_qa_studies_family",
        "bench_tomato_qa_studies_family",
    ],
)
def test_benches_w1452(fam):
    out = getattr(benches_w1452, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
