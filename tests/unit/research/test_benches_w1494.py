import pytest

from quant_fund.research import benches_w1494


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anole_qa_studies_family",
        "bench_chameleon_qa_studies_family",
        "bench_hognose_qa_studies_family",
        "bench_skink_qa_studies_family",
        "bench_terrapin_qa_studies_family",
        "bench_tuatara_qa_studies_family",
    ],
)
def test_benches_w1494(fam):
    out = getattr(benches_w1494, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
