import pytest

from quant_fund.research import benches_w1818


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bastet2_qa_studies_family",
        "bench_geb2_qa_studies_family",
        "bench_hathor2_qa_studies_family",
        "bench_nut2_qa_studies_family",
        "bench_sekhmet2_qa_studies_family",
        "bench_tefnut2_qa_studies_family",
    ],
)
def test_benches_w1818(fam):
    out = getattr(benches_w1818, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
