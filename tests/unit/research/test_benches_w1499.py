import pytest

from quant_fund.research import benches_w1499


@pytest.mark.parametrize(
    "fam",
    [
        "bench_oregano_qa_studies_family",
        "bench_parsley_qa_studies_family",
        "bench_rosemary_qa_studies_family",
        "bench_saffron_qa_studies_family",
        "bench_tarragon_qa_studies_family",
        "bench_turmeric_qa_studies_family",
    ],
)
def test_benches_w1499(fam):
    out = getattr(benches_w1499, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
