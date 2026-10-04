import pytest

from quant_fund.research import benches_w1389


@pytest.mark.parametrize(
    "fam",
    [
        "bench_book_sum_studies_family",
        "bench_fanout_qa_studies_family",
        "bench_infinitesum_studies_family",
        "bench_marlense_studies_family",
        "bench_narra_sum_studies_family",
        "bench_quote_sum_studies_family",
    ],
)
def test_benches_w1389(fam):
    out = getattr(benches_w1389, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
