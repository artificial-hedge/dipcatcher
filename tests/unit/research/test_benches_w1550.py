import pytest

from quant_fund.research import benches_w1550


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agama_qa_studies_family",
        "bench_chuckwalla_qa_studies_family",
        "bench_frilled_lizard_qa_studies_family",
        "bench_monitor_lizard_qa_studies_family",
        "bench_tegu_qa_studies_family",
        "bench_uromastyx_qa_studies_family",
    ],
)
def test_benches_w1550(fam):
    out = getattr(benches_w1550, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
