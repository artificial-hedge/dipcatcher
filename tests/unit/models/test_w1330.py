import pytest


@pytest.mark.parametrize(
    "name",
    [
        "safe_rlhf_studies",
        "beaver_safe_studies",
        "sos_bench_studies",
        "do_not_answer_studies",
        "honest_eval_studies",
        "hh_rlhf_studies",
    ],
)
def test_w1330_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
