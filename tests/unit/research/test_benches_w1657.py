import pytest

from quant_fund.research import benches_w1657


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gargoyle_qa_studies_family",
        "bench_guivre_qa_studies_family",
        "bench_melusine_qa_studies_family",
        "bench_quinotaur_qa_studies_family",
        "bench_tarascon_qa_studies_family",
        "bench_tarrasque_qa_studies_family",
    ],
)
def test_benches_w1657(fam):
    out = getattr(benches_w1657, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
