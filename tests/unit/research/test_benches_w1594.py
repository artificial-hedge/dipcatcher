import pytest

from quant_fund.research import benches_w1594


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bonobo_qa_studies_family",
        "bench_chimpanzee_qa_studies_family",
        "bench_douc_qa_studies_family",
        "bench_proboscis_qa_studies_family",
        "bench_siamang_qa_studies_family",
        "bench_snub_nosed_qa_studies_family",
    ],
)
def test_benches_w1594(fam):
    out = getattr(benches_w1594, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
