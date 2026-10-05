import pytest

from quant_fund.research import benches_w1811


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bes2_qa_studies_family",
        "bench_geb2_qa_studies_family",
        "bench_nephthys2_qa_studies_family",
        "bench_ptah2_qa_studies_family",
        "bench_sekhmet2_qa_studies_family",
        "bench_thoth2_qa_studies_family",
    ],
)
def test_benches_w1811(fam):
    out = getattr(benches_w1811, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
