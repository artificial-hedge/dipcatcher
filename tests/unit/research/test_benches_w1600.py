import pytest

from quant_fund.research import benches_w1600


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abalone_qa_studies_family",
        "bench_chiton_qa_studies_family",
        "bench_cockle_qa_studies_family",
        "bench_cowrie_qa_studies_family",
        "bench_limpet_qa_studies_family",
        "bench_periwinkle_qa_studies_family",
    ],
)
def test_benches_w1600(fam):
    out = getattr(benches_w1600, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
