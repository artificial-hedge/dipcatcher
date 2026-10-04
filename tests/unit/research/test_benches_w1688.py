import pytest

from quant_fund.research import benches_w1688


@pytest.mark.parametrize(
    "fam",
    [
        "bench_indiges_qa_studies_family",
        "bench_lar_qa_studies_family",
        "bench_numen_qa_studies_family",
        "bench_penates_qa_studies_family",
        "bench_terminus_qa_studies_family",
        "bench_vertumnus_qa_studies_family",
    ],
)
def test_benches_w1688(fam):
    out = getattr(benches_w1688, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
