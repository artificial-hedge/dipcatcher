import pytest

from quant_fund.research import benches_w1744


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agloolik_qa_studies_family",
        "bench_aumanil_qa_studies_family",
        "bench_nuktessien_qa_studies_family",
        "bench_sedna_qa_studies_family",
        "bench_tekkeitsertok_qa_studies_family",
        "bench_torngarsuk_qa_studies_family",
    ],
)
def test_benches_w1744(fam):
    out = getattr(benches_w1744, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
