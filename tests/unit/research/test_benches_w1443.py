import pytest

from quant_fund.research import benches_w1443


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amber_qa_studies_family",
        "bench_amethyst_qa_studies_family",
        "bench_crystal_qa_studies_family",
        "bench_diamond_qa_studies_family",
        "bench_emerald_qa_studies_family",
        "bench_jade_qa_studies_family",
    ],
)
def test_benches_w1443(fam):
    out = getattr(benches_w1443, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
