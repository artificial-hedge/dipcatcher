import pytest

from quant_fund.research import benches_w1921


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baubas_qa_studies_family",
        "bench_kaukas_qa_studies_family",
        "bench_lauma_qa_studies_family",
        "bench_pukis_qa_studies_family",
        "bench_spigana_qa_studies_family",
        "bench_vilkacis_qa_studies_family",
    ],
)
def test_benches_w1921(fam):
    out = getattr(benches_w1921, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
