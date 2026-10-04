import pytest

from quant_fund.research import benches_w1644


@pytest.mark.parametrize(
    "fam",
    [
        "bench_charybdis_qa_studies_family",
        "bench_cyclops_2_qa_studies_family",
        "bench_hydra_3_qa_studies_family",
        "bench_medusa_2_qa_studies_family",
        "bench_scylla_qa_studies_family",
        "bench_siren_2_qa_studies_family",
    ],
)
def test_benches_w1644(fam):
    out = getattr(benches_w1644, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
