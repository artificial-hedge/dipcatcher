import pytest

from quant_fund.research import benches_w1836


@pytest.mark.parametrize(
    "fam",
    [
        "bench_axom2_qa_studies_family",
        "bench_chrysaoreus2_qa_studies_family",
        "bench_hekate2_qa_studies_family",
        "bench_labrandeus2_qa_studies_family",
        "bench_panamara2_qa_studies_family",
        "bench_stratios2_qa_studies_family",
    ],
)
def test_benches_w1836(fam):
    out = getattr(benches_w1836, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
