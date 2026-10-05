import pytest

from quant_fund.research import benches_w1797


@pytest.mark.parametrize(
    "fam",
    [
        "bench_atropos_qa_studies_family",
        "bench_dikaion_qa_studies_family",
        "bench_eunomia_qa_studies_family",
        "bench_lachesis_qa_studies_family",
        "bench_metis2_qa_studies_family",
        "bench_peitho_qa_studies_family",
    ],
)
def test_benches_w1797(fam):
    out = getattr(benches_w1797, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
