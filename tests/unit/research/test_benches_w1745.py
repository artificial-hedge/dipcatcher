import pytest

from quant_fund.research import benches_w1745


@pytest.mark.parametrize(
    "fam",
    [
        "bench_altjira_qa_studies_family",
        "bench_bunyip_qa_studies_family",
        "bench_mimis_qa_studies_family",
        "bench_rainbow_serpent_qa_studies_family",
        "bench_wandjina_qa_studies_family",
        "bench_yowie_qa_studies_family",
    ],
)
def test_benches_w1745(fam):
    out = getattr(benches_w1745, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
