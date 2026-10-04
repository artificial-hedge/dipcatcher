import pytest

from quant_fund.research import benches_w1771


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hannahanna_qa_studies_family",
        "bench_illuyanka_qa_studies_family",
        "bench_inara_qa_studies_family",
        "bench_kamrusepa_qa_studies_family",
        "bench_tarhunna_qa_studies_family",
        "bench_telepinus_qa_studies_family",
    ],
)
def test_benches_w1771(fam):
    out = getattr(benches_w1771, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
