import pytest

from quant_fund.research import benches_w1748


@pytest.mark.parametrize(
    "fam",
    [
        "bench_beg_tse_qa_studies_family",
        "bench_dorje_legpa_qa_studies_family",
        "bench_palden_lhamo_qa_studies_family",
        "bench_pehar_qa_studies_family",
        "bench_tsen_god_qa_studies_family",
        "bench_tsiu_marpo_qa_studies_family",
    ],
)
def test_benches_w1748(fam):
    out = getattr(benches_w1748, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
