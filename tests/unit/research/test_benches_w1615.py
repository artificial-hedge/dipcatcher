import pytest

from quant_fund.research import benches_w1615


@pytest.mark.parametrize(
    "fam",
    [
        "bench_golden_brown_qa_studies_family",
        "bench_gray_mouse_qa_studies_family",
        "bench_pygmy_qa_studies_family",
        "bench_slender_qa_studies_family",
        "bench_slow_qa_studies_family",
        "bench_thin_spined_qa_studies_family",
    ],
)
def test_benches_w1615(fam):
    out = getattr(benches_w1615, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
