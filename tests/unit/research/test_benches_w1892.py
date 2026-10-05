import pytest

from quant_fund.research import benches_w1892


@pytest.mark.parametrize(
    "fam",
    [
        "bench_al_basti_qa_studies_family",
        "bench_ananke_libya_qa_studies_family",
        "bench_encantado_qa_studies_family",
        "bench_mithra_iran_qa_studies_family",
        "bench_mitra_persian_qa_studies_family",
        "bench_perangal_qa_studies_family",
    ],
)
def test_benches_w1892(fam):
    out = getattr(benches_w1892, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
