import pytest

from quant_fund.research import benches_w1518


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coquette_qa_studies_family",
        "bench_fairy_qa_studies_family",
        "bench_jacobin_qa_studies_family",
        "bench_lancebill_qa_studies_family",
        "bench_sabrewing_qa_studies_family",
        "bench_sheartail_qa_studies_family",
    ],
)
def test_benches_w1518(fam):
    out = getattr(benches_w1518, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
