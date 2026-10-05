import pytest

from quant_fund.research import benches_w1853


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amesemi_qa_studies_family",
        "bench_apedemak_qa_studies_family",
        "bench_aresnuphis_qa_studies_family",
        "bench_dedun_qa_studies_family",
        "bench_sabios_qa_studies_family",
        "bench_sebiumeker_qa_studies_family",
    ],
)
def test_benches_w1853(fam):
    out = getattr(benches_w1853, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
