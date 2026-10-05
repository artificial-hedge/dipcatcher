import pytest

from quant_fund.research import benches_w1848


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apedemak_qa_studies_family",
        "bench_arensnuphis_qa_studies_family",
        "bench_dedwen_qa_studies_family",
        "bench_mandulis_qa_studies_family",
        "bench_miket_qa_studies_family",
        "bench_sebiumeker_qa_studies_family",
    ],
)
def test_benches_w1848(fam):
    out = getattr(benches_w1848, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
