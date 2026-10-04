import pytest

from quant_fund.research import benches_w1458


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arroyo_qa_studies_family",
        "bench_butte_qa_studies_family",
        "bench_camel_qa_studies_family",
        "bench_caravan_qa_studies_family",
        "bench_mirage_qa_studies_family",
        "bench_oasis_qa_studies_family",
    ],
)
def test_benches_w1458(fam):
    out = getattr(benches_w1458, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
