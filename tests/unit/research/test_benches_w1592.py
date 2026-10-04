import pytest

from quant_fund.research import benches_w1592


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bushbaby_qa_studies_family",
        "bench_galago_qa_studies_family",
        "bench_indri_qa_studies_family",
        "bench_loris_qa_studies_family",
        "bench_potto_qa_studies_family",
        "bench_tarsier_qa_studies_family",
    ],
)
def test_benches_w1592(fam):
    out = getattr(benches_w1592, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
