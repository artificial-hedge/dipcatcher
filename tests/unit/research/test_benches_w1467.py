import pytest

from quant_fund.research import benches_w1467


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arch_qa_studies_family",
        "bench_steppe_qa_studies_family",
        "bench_summit_qa_studies_family",
        "bench_tundra_qa_studies_family",
        "bench_valley_qa_studies_family",
        "bench_volcano_qa_studies_family",
    ],
)
def test_benches_w1467(fam):
    out = getattr(benches_w1467, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
