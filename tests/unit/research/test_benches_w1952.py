import pytest

from quant_fund.research import benches_w1952


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alloces_qa_studies_family",
        "bench_balam_qa_studies_family",
        "bench_camio_qa_studies_family",
        "bench_foras_qa_studies_family",
        "bench_furcas_qa_studies_family",
        "bench_gaap_qa_studies_family",
    ],
)
def test_benches_w1952(fam):
    out = getattr(benches_w1952, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
