import pytest

from quant_fund.research import benches_w1284


@pytest.mark.parametrize(
    "fam",
    [
        "bench_citation_check_studies_family",
        "bench_claim_verifier_studies_family",
        "bench_entailment_studies_family",
        "bench_factuality_score_studies_family",
        "bench_grounding_verify_studies_family",
        "bench_self_reflect_studies_family",
    ],
)
def test_benches_w1284(fam):
    out = getattr(benches_w1284, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
