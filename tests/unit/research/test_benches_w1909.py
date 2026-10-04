import pytest

from quant_fund.research import benches_w1909


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aeshma_qa_studies_family",
        "bench_astwihad_qa_studies_family",
        "bench_azhi_dahaka_qa_studies_family",
        "bench_druj_qa_studies_family",
        "bench_jahi_qa_studies_family",
        "bench_nasu_qa_studies_family",
    ],
)
def test_benches_w1909(fam):
    out = getattr(benches_w1909, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
