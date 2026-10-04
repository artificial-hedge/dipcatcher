import pytest

from quant_fund.research import benches_w1576


@pytest.mark.parametrize(
    "fam",
    [
        "bench_binturong_qa_studies_family",
        "bench_fossa_qa_studies_family",
        "bench_honey_badger_qa_studies_family",
        "bench_kusimanse_qa_studies_family",
        "bench_maned_wolf_qa_studies_family",
        "bench_sun_bear_qa_studies_family",
    ],
)
def test_benches_w1576(fam):
    out = getattr(benches_w1576, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
