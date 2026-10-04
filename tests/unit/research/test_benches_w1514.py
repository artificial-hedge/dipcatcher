import pytest

from quant_fund.research import benches_w1514


@pytest.mark.parametrize(
    "fam",
    [
        "bench_blue_morpho_qa_studies_family",
        "bench_cabbage_white_qa_studies_family",
        "bench_fritillary_qa_studies_family",
        "bench_monarch_qa_studies_family",
        "bench_painted_lady_qa_studies_family",
        "bench_swallowtail_qa_studies_family",
    ],
)
def test_benches_w1514(fam):
    out = getattr(benches_w1514, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
