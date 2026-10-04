import pytest

from quant_fund.research import benches_w1676


@pytest.mark.parametrize(
    "fam",
    [
        "bench_kinnara_qa_studies_family",
        "bench_pisacha_qa_studies_family",
        "bench_uraga_qa_studies_family",
        "bench_vetala_qa_studies_family",
        "bench_vidyadhara_qa_studies_family",
        "bench_yakshini_qa_studies_family",
    ],
)
def test_benches_w1676(fam):
    out = getattr(benches_w1676, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
