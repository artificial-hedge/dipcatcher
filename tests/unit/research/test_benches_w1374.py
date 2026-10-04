import pytest

from quant_fund.research import benches_w1374


@pytest.mark.parametrize(
    "fam",
    [
        "bench_deduc_lite_studies_family",
        "bench_entail_bank_studies_family",
        "bench_folio_lite_studies_family",
        "bench_logic_nli_studies_family",
        "bench_proof_writer_studies_family",
        "bench_rule_taker_studies_family",
    ],
)
def test_benches_w1374(fam):
    out = getattr(benches_w1374, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
