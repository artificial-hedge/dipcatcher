import pytest

from quant_fund.research import benches_w1689


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apsara_qa_studies_family",
        "bench_bhairava_qa_studies_family",
        "bench_bhuta_qa_studies_family",
        "bench_pretas_qa_studies_family",
        "bench_vetal_qa_studies_family",
        "bench_yaksha_qa_studies_family",
    ],
)
def test_benches_w1689(fam):
    out = getattr(benches_w1689, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
