import pytest

from quant_fund.research import benches_w1833


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anahit2_qa_studies_family",
        "bench_aramazd2_qa_studies_family",
        "bench_astghik2_qa_studies_family",
        "bench_mher2_qa_studies_family",
        "bench_tir2_qa_studies_family",
        "bench_vahagn2_qa_studies_family",
    ],
)
def test_benches_w1833(fam):
    out = getattr(benches_w1833, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
