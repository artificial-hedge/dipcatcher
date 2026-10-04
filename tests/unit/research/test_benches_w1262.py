import pytest

from quant_fund.research import benches_w1262


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dynamic_borrowing_studies_family",
        "bench_e_value_studies_family",
        "bench_master_protocol_studies_family",
        "bench_stepped_wedge_studies_family",
        "bench_target_trial_emulation_studies_family",
        "bench_win_ratio_studies_family",
    ],
)
def test_benches_w1262(fam):
    out = getattr(benches_w1262, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
