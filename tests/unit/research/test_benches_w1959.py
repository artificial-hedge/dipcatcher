import pytest

from quant_fund.research import benches_w1959


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adramelech_qa_studies_family",
        "bench_azazel_qa_studies_family",
        "bench_belphegor_qa_studies_family",
        "bench_ipos_qa_studies_family",
        "bench_marchosias_qa_studies_family",
        "bench_phenex_qa_studies_family",
    ],
)
def test_benches_w1959(fam):
    out = getattr(benches_w1959, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
