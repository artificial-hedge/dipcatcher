import pytest

from quant_fund.research import benches_w1911


@pytest.mark.parametrize(
    "fam",
    [
        "bench_divsalar_qa_studies_family",
        "bench_khrafstra_qa_studies_family",
        "bench_leshenka_qa_studies_family",
        "bench_pari_vatra_qa_studies_family",
        "bench_srosh_demon_qa_studies_family",
        "bench_urvan_qa_studies_family",
    ],
)
def test_benches_w1911(fam):
    out = getattr(benches_w1911, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
