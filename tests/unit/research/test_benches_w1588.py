import pytest

from quant_fund.research import benches_w1588


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bearded_seal_qa_studies_family",
        "bench_crabeater_qa_studies_family",
        "bench_hooded_seal_qa_studies_family",
        "bench_ribbon_seal_qa_studies_family",
        "bench_ringed_seal_qa_studies_family",
        "bench_ross_seal_qa_studies_family",
    ],
)
def test_benches_w1588(fam):
    out = getattr(benches_w1588, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
