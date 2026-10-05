import pytest

from quant_fund.research import benches_w1916


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alp_qa_studies_family",
        "bench_doppelganger_qa_studies_family",
        "bench_kobold_qa_studies_family",
        "bench_mahr_qa_studies_family",
        "bench_poltergeist_qa_studies_family",
        "bench_tatzelwurm_qa_studies_family",
    ],
)
def test_benches_w1916(fam):
    out = getattr(benches_w1916, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
