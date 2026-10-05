import pytest

from quant_fund.research import benches_w1919


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bodach_qa_studies_family",
        "bench_caointeach_qa_studies_family",
        "bench_fachan_qa_studies_family",
        "bench_geancanach_qa_studies_family",
        "bench_kilmoulis_qa_studies_family",
        "bench_shellycoat_qa_studies_family",
    ],
)
def test_benches_w1919(fam):
    out = getattr(benches_w1919, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
