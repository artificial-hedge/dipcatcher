import pytest

from quant_fund.research import benches_w1948


@pytest.mark.parametrize(
    "fam",
    [
        "bench_asmodeus_qa_studies_family",
        "bench_astaroth_qa_studies_family",
        "bench_belial_qa_studies_family",
        "bench_furfur_qa_studies_family",
        "bench_paimon_qa_studies_family",
        "bench_stolas_qa_studies_family",
    ],
)
def test_benches_w1948(fam):
    out = getattr(benches_w1948, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
