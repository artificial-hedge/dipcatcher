import pytest

from quant_fund.research import benches_w1791


@pytest.mark.parametrize(
    "fam",
    [
        "bench_frey_qa_studies_family",
        "bench_freya2_qa_studies_family",
        "bench_magni_qa_studies_family",
        "bench_modi_qa_studies_family",
        "bench_njord_qa_studies_family",
        "bench_tyr2_qa_studies_family",
    ],
)
def test_benches_w1791(fam):
    out = getattr(benches_w1791, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
