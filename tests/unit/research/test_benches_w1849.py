import pytest

from quant_fund.research import benches_w1849


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alouq_qa_studies_family",
        "bench_aster2_qa_studies_family",
        "bench_beher_qa_studies_family",
        "bench_mahrem_qa_studies_family",
        "bench_medr_qa_studies_family",
        "bench_ruda_qa_studies_family",
    ],
)
def test_benches_w1849(fam):
    out = getattr(benches_w1849, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
