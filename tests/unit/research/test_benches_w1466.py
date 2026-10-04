import pytest

from quant_fund.research import benches_w1466


@pytest.mark.parametrize(
    "fam",
    [
        "bench_basalt_qa_studies_family",
        "bench_cathedral_qa_studies_family",
        "bench_chasm_qa_studies_family",
        "bench_crag_qa_studies_family",
        "bench_plateau_qa_studies_family",
        "bench_ravine_qa_studies_family",
    ],
)
def test_benches_w1466(fam):
    out = getattr(benches_w1466, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
