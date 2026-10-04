import pytest

from quant_fund.research import benches_w1729


@pytest.mark.parametrize(
    "fam",
    [
        "bench_austra_qa_studies_family",
        "bench_jumis_qa_studies_family",
        "bench_laima_qa_studies_family",
        "bench_laume_qa_studies_family",
        "bench_perkunas_qa_studies_family",
        "bench_zemyna_qa_studies_family",
    ],
)
def test_benches_w1729(fam):
    out = getattr(benches_w1729, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
