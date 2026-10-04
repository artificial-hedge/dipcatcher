import pytest

from quant_fund.research import benches_w1762


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baldr_qa_studies_family",
        "bench_bragi_qa_studies_family",
        "bench_freya_qa_studies_family",
        "bench_heimdall_qa_studies_family",
        "bench_idunn_qa_studies_family",
        "bench_tyr_qa_studies_family",
    ],
)
def test_benches_w1762(fam):
    out = getattr(benches_w1762, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
