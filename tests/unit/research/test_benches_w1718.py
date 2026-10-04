import pytest

from quant_fund.research import benches_w1718


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fufluns_qa_studies_family",
        "bench_menrva_qa_studies_family",
        "bench_tinia_qa_studies_family",
        "bench_turan_qa_studies_family",
        "bench_veltha_qa_studies_family",
        "bench_voltumna_qa_studies_family",
    ],
)
def test_benches_w1718(fam):
    out = getattr(benches_w1718, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
