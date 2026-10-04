import pytest

from quant_fund.research import benches_w1683


@pytest.mark.parametrize(
    "fam",
    [
        "bench_danava_qa_studies_family",
        "bench_gana_qa_studies_family",
        "bench_gandharva_qa_studies_family",
        "bench_kalakeya_qa_studies_family",
        "bench_kimpurusha_qa_studies_family",
        "bench_rakshasa_qa_studies_family",
    ],
)
def test_benches_w1683(fam):
    out = getattr(benches_w1683, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
