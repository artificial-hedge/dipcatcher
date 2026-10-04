import pytest

from quant_fund.research import benches_w1928


@pytest.mark.parametrize(
    "fam",
    [
        "bench_kehua_qa_studies_family",
        "bench_maero_qa_studies_family",
        "bench_ngarara_qa_studies_family",
        "bench_patupaiarehe_qa_studies_family",
        "bench_ponaturi_qa_studies_family",
        "bench_taipo_qa_studies_family",
    ],
)
def test_benches_w1928(fam):
    out = getattr(benches_w1928, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
