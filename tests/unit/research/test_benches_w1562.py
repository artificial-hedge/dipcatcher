import pytest

from quant_fund.research import benches_w1562


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boxfish_qa_studies_family",
        "bench_clownfish_qa_studies_family",
        "bench_dragonet_qa_studies_family",
        "bench_mandarinfish_qa_studies_family",
        "bench_pipefish_qa_studies_family",
        "bench_pufferfish_qa_studies_family",
    ],
)
def test_benches_w1562(fam):
    out = getattr(benches_w1562, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
