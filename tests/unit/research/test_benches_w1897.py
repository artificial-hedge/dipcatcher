import pytest

from quant_fund.research import benches_w1897


@pytest.mark.parametrize(
    "fam",
    [
        "bench_lamia_qa_studies_family",
        "bench_mormo_qa_studies_family",
        "bench_nachzehrer_qa_studies_family",
        "bench_striga_qa_studies_family",
        "bench_strix_qa_studies_family",
        "bench_vrykolakas_qa_studies_family",
    ],
)
def test_benches_w1897(fam):
    out = getattr(benches_w1897, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
