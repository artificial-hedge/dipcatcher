import pytest

from quant_fund.research import benches_w1944


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cheonyeo_gwishin_qa_studies_family",
        "bench_dokkaebi_qa_studies_family",
        "bench_gumiho_qa_studies_family",
        "bench_gwishin_qa_studies_family",
        "bench_mul_gwishin_qa_studies_family",
        "bench_oeggwi_qa_studies_family",
    ],
)
def test_benches_w1944(fam):
    out = getattr(benches_w1944, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
