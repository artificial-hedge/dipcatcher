import pytest

from quant_fund.research import benches_w1681


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anansi_qa_studies_family",
        "bench_impundulu_qa_studies_family",
        "bench_kalulu_qa_studies_family",
        "bench_mamiwata_qa_studies_family",
        "bench_sasabonsam_qa_studies_family",
        "bench_tokoloshe_qa_studies_family",
    ],
)
def test_benches_w1681(fam):
    out = getattr(benches_w1681, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
