import pytest

from quant_fund.research import benches_w1503


@pytest.mark.parametrize(
    "fam",
    [
        "bench_garter_qa_studies_family",
        "bench_keelback_qa_studies_family",
        "bench_kingsnake_qa_studies_family",
        "bench_mockviper_qa_studies_family",
        "bench_racer_qa_studies_family",
        "bench_sidewinder_qa_studies_family",
    ],
)
def test_benches_w1503(fam):
    out = getattr(benches_w1503, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
