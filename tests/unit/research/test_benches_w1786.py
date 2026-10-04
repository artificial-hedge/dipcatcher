import pytest

from quant_fund.research import benches_w1786


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ea_qa_studies_family",
        "bench_humbaba_qa_studies_family",
        "bench_pazuzu_qa_studies_family",
        "bench_sargon_qa_studies_family",
        "bench_semiramis_qa_studies_family",
        "bench_utnapishtim_qa_studies_family",
    ],
)
def test_benches_w1786(fam):
    out = getattr(benches_w1786, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
