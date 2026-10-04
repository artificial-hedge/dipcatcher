import pytest

from quant_fund.research import benches_w1699


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amaterasu_qa_studies_family",
        "bench_hachiman_qa_studies_family",
        "bench_inari_qa_studies_family",
        "bench_raijin_qa_studies_family",
        "bench_susanoo_qa_studies_family",
        "bench_tsukuyomi_qa_studies_family",
    ],
)
def test_benches_w1699(fam):
    out = getattr(benches_w1699, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
