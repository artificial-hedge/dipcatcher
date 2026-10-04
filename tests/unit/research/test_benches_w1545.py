import pytest

from quant_fund.research import benches_w1545


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amazon_qa_studies_family",
        "bench_cockatoo_qa_studies_family",
        "bench_conure_qa_studies_family",
        "bench_kakapo_qa_studies_family",
        "bench_kea_qa_studies_family",
        "bench_lorikeet_qa_studies_family",
    ],
)
def test_benches_w1545(fam):
    out = getattr(benches_w1545, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
