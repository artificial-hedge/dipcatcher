import pytest

from quant_fund.research import benches_w1679


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alfheim_qa_studies_family",
        "bench_bergrisi_qa_studies_family",
        "bench_geirahod_qa_studies_family",
        "bench_huldra_qa_studies_family",
        "bench_troll_qa_studies_family",
        "bench_vaetter_qa_studies_family",
    ],
)
def test_benches_w1679(fam):
    out = getattr(benches_w1679, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
