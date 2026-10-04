import pytest

from quant_fund.research import benches_w1781


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apollo_qa_studies_family",
        "bench_artemis_qa_studies_family",
        "bench_athena_qa_studies_family",
        "bench_demeter_qa_studies_family",
        "bench_hera_qa_studies_family",
        "bench_persephone_qa_studies_family",
    ],
)
def test_benches_w1781(fam):
    out = getattr(benches_w1781, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
