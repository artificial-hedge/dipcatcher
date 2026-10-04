import pytest

from quant_fund.research import benches_w1858


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aracus_qa_studies_family",
        "bench_cosus_qa_studies_family",
        "bench_cronia_qa_studies_family",
        "bench_munidis_qa_studies_family",
        "bench_quangeio_qa_studies_family",
        "bench_reo_qa_studies_family",
    ],
)
def test_benches_w1858(fam):
    out = getattr(benches_w1858, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
