import pytest

from quant_fund.research import benches_w1542


@pytest.mark.parametrize(
    "fam",
    [
        "bench_accipiter_qa_studies_family",
        "bench_bateleur_qa_studies_family",
        "bench_falconet_qa_studies_family",
        "bench_harpy_qa_studies_family",
        "bench_lammergeier_qa_studies_family",
        "bench_seriema_qa_studies_family",
    ],
)
def test_benches_w1542(fam):
    out = getattr(benches_w1542, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
