import pytest

from quant_fund.research import benches_w1670


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alseid_qa_studies_family",
        "bench_gnome_volk_qa_studies_family",
        "bench_meliae_qa_studies_family",
        "bench_napaea_qa_studies_family",
        "bench_oread_qa_studies_family",
        "bench_sylph_qa_studies_family",
    ],
)
def test_benches_w1670(fam):
    out = getattr(benches_w1670, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
