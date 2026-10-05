import pytest

from quant_fund.research import benches_w1812


@pytest.mark.parametrize(
    "fam",
    [
        "bench_centeotl2_qa_studies_family",
        "bench_mayahuel2_qa_studies_family",
        "bench_mixcoatl2_qa_studies_family",
        "bench_tlaloc2_qa_studies_family",
        "bench_xipe2_qa_studies_family",
        "bench_xochipilli2_qa_studies_family",
    ],
)
def test_benches_w1812(fam):
    out = getattr(benches_w1812, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
