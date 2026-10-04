import pytest

from quant_fund.research import benches_w1662


@pytest.mark.parametrize(
    "fam",
    [
        "bench_banshee_qa_studies_family",
        "bench_dullahan_qa_studies_family",
        "bench_kelpie_qa_studies_family",
        "bench_leprechaun_qa_studies_family",
        "bench_puca_qa_studies_family",
        "bench_selkie_qa_studies_family",
    ],
)
def test_benches_w1662(fam):
    out = getattr(benches_w1662, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
