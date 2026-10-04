import pytest

from quant_fund.research import benches_w1408


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abduct_qa_studies_family",
        "bench_analogy_qa_studies_family",
        "bench_arct_lite_studies_family",
        "bench_entailment_qa_studies_family",
        "bench_fusion_qa_studies_family",
        "bench_proof_qa_studies_family",
    ],
)
def test_benches_w1408(fam):
    out = getattr(benches_w1408, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
