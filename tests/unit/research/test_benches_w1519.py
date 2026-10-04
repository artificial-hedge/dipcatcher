import pytest

from quant_fund.research import benches_w1519


@pytest.mark.parametrize(
    "fam",
    [
        "bench_empusa_qa_studies_family",
        "bench_ghost_mantis_qa_studies_family",
        "bench_mantidfly_qa_studies_family",
        "bench_orchid_mantis_qa_studies_family",
        "bench_praying_mantis_qa_studies_family",
        "bench_shield_mantis_qa_studies_family",
    ],
)
def test_benches_w1519(fam):
    out = getattr(benches_w1519, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
