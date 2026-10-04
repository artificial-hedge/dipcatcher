import pytest

from quant_fund.research import benches_w1484


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bandicoot_qa_studies_family",
        "bench_koala_qa_studies_family",
        "bench_numbat_qa_studies_family",
        "bench_quokka_qa_studies_family",
        "bench_wallaby_qa_studies_family",
        "bench_wombat_qa_studies_family",
    ],
)
def test_benches_w1484(fam):
    out = getattr(benches_w1484, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
