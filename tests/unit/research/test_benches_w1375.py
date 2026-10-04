import pytest

from quant_fund.research import benches_w1375


@pytest.mark.parametrize(
    "fam",
    [
        "bench_facet_sum_studies_family",
        "bench_ms2_lite_studies_family",
        "bench_patent_sum_studies_family",
        "bench_sci_lay_studies_family",
        "bench_scitldr_lite_studies_family",
        "bench_spectrum_sum_studies_family",
    ],
)
def test_benches_w1375(fam):
    out = getattr(benches_w1375, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
