import pytest

from quant_fund.research import benches_w1563


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cleaner_shrimp_qa_studies_family",
        "bench_decorator_crab_qa_studies_family",
        "bench_hermit_crab_qa_studies_family",
        "bench_mantis_shrimp_qa_studies_family",
        "bench_pistol_shrimp_qa_studies_family",
        "bench_porcelain_crab_qa_studies_family",
    ],
)
def test_benches_w1563(fam):
    out = getattr(benches_w1563, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
