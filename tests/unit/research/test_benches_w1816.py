import pytest

from quant_fund.research import benches_w1816


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fujin2_qa_studies_family",
        "bench_hachiman2_qa_studies_family",
        "bench_inari2_qa_studies_family",
        "bench_raijin2_qa_studies_family",
        "bench_susanoo2_qa_studies_family",
        "bench_tsukuyomi2_qa_studies_family",
    ],
)
def test_benches_w1816(fam):
    out = getattr(benches_w1816, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
