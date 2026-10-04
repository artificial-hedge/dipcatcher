import pytest

from quant_fund.research import benches_w1532


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bee_eater_qa_studies_family",
        "bench_jacamar_qa_studies_family",
        "bench_kookaburra_qa_studies_family",
        "bench_motmot_qa_studies_family",
        "bench_roller_qa_studies_family",
        "bench_tody_qa_studies_family",
    ],
)
def test_benches_w1532(fam):
    out = getattr(benches_w1532, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
