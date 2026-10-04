import pytest

from quant_fund.research import benches_w1927


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amautalik_qa_studies_family",
        "bench_ijiraq_qa_studies_family",
        "bench_mahaha_qa_studies_family",
        "bench_qivittoq_qa_studies_family",
        "bench_tornit_qa_studies_family",
        "bench_tupilaq_qa_studies_family",
    ],
)
def test_benches_w1927(fam):
    out = getattr(benches_w1927, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
