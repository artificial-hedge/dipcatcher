import pytest

from quant_fund.research import benches_w1878


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baal_marod_qa_studies_family",
        "bench_bozrum_qa_studies_family",
        "bench_guillyn_qa_studies_family",
        "bench_hammonites_qa_studies_family",
        "bench_weded_qa_studies_family",
        "bench_yamenna_qa_studies_family",
    ],
)
def test_benches_w1878(fam):
    out = getattr(benches_w1878, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
