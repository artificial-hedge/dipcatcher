import pytest

from quant_fund.research import benches_w1792


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agga_qa_studies_family",
        "bench_babbar_qa_studies_family",
        "bench_enmerkar2_qa_studies_family",
        "bench_lugulbanda_qa_studies_family",
        "bench_ninsun_qa_studies_family",
        "bench_urukagina_qa_studies_family",
    ],
)
def test_benches_w1792(fam):
    out = getattr(benches_w1792, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
