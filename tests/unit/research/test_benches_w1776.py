import pytest

from quant_fund.research import benches_w1776


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bari_qa_studies_family",
        "bench_kongjwi_qa_studies_family",
        "bench_ondal_qa_studies_family",
        "bench_pyonggang_qa_studies_family",
        "bench_samshin_qa_studies_family",
        "bench_shimchong_qa_studies_family",
    ],
)
def test_benches_w1776(fam):
    out = getattr(benches_w1776, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
