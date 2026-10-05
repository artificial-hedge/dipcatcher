import pytest

from quant_fund.research import benches_w1793


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cluentia_qa_studies_family",
        "bench_faunus_qa_studies_family",
        "bench_larunda_qa_studies_family",
        "bench_mutina_qa_studies_family",
        "bench_quirinus_qa_studies_family",
        "bench_tellus_qa_studies_family",
    ],
)
def test_benches_w1793(fam):
    out = getattr(benches_w1793, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
