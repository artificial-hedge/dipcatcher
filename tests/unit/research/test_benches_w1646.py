import pytest

from quant_fund.research import benches_w1646


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fenrir_2_qa_studies_family",
        "bench_garm_qa_studies_family",
        "bench_jormungandr_qa_studies_family",
        "bench_nidhogg_qa_studies_family",
        "bench_ratatoskr_qa_studies_family",
        "bench_sleipnir_qa_studies_family",
    ],
)
def test_benches_w1646(fam):
    out = getattr(benches_w1646, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
