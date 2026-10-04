import pytest

from quant_fund.research import benches_w1893


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alu_demon_qa_studies_family",
        "bench_ardat_lili_qa_studies_family",
        "bench_galla_demon_qa_studies_family",
        "bench_lamashtu_qa_studies_family",
        "bench_lilitu_qa_studies_family",
        "bench_rabisu_qa_studies_family",
    ],
)
def test_benches_w1893(fam):
    out = getattr(benches_w1893, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
