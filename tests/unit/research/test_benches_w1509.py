import pytest

from quant_fund.research import benches_w1509


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bunting_qa_studies_family",
        "bench_grosbeak_qa_studies_family",
        "bench_nuthatch_qa_studies_family",
        "bench_tanager_qa_studies_family",
        "bench_titmouse_qa_studies_family",
        "bench_vireo_qa_studies_family",
    ],
)
def test_benches_w1509(fam):
    out = getattr(benches_w1509, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
