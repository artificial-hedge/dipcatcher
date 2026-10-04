import pytest

from quant_fund.research import benches_w1898


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bajang_qa_studies_family",
        "bench_kum_kum_qa_studies_family",
        "bench_pelesit_qa_studies_family",
        "bench_penanggalan_qa_studies_family",
        "bench_pontianak_qa_studies_family",
        "bench_toyol_qa_studies_family",
    ],
)
def test_benches_w1898(fam):
    out = getattr(benches_w1898, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
