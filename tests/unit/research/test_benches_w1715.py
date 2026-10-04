import pytest

from quant_fund.research import benches_w1715


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bendis_qa_studies_family",
        "bench_darzalas_qa_studies_family",
        "bench_derzelas_qa_studies_family",
        "bench_kezion_qa_studies_family",
        "bench_sabazios_qa_studies_family",
        "bench_zamolxis_qa_studies_family",
    ],
)
def test_benches_w1715(fam):
    out = getattr(benches_w1715, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
