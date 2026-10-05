import pytest

from quant_fund.research import benches_w1937


@pytest.mark.parametrize(
    "fam",
    [
        "bench_caleuche_qa_studies_family",
        "bench_camahueto_qa_studies_family",
        "bench_fiura_qa_studies_family",
        "bench_invunche_qa_studies_family",
        "bench_pincoya_qa_studies_family",
        "bench_trauco_qa_studies_family",
    ],
)
def test_benches_w1937(fam):
    out = getattr(benches_w1937, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
