import pytest

from quant_fund.research import benches_w1622


@pytest.mark.parametrize(
    "fam",
    [
        "bench_snowcock_qa_studies_family",
        "bench_barbary_qa_studies_family",
        "bench_blue_sheep_qa_studies_family",
        "bench_himalayan_qa_studies_family",
        "bench_nilgiri_qa_studies_family",
        "bench_snow_leopard_qa_studies_family",
    ],
)
def test_benches_w1622(fam):
    out = getattr(benches_w1622, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
