import pytest

from quant_fund.research import benches_w1546


@pytest.mark.parametrize(
    "fam",
    [
        "bench_corncrake_qa_studies_family",
        "bench_flufftail_qa_studies_family",
        "bench_sora_qa_studies_family",
        "bench_sungrebe_qa_studies_family",
        "bench_swamphen_qa_studies_family",
        "bench_takhe_qa_studies_family",
    ],
)
def test_benches_w1546(fam):
    out = getattr(benches_w1546, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
