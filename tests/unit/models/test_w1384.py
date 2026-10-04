import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aider_polyglot_studies",
        "mbti_eval_studies",
        "livebench_arena_studies",
        "olmes_lite_studies",
        "plus_eval_studies",
        "hum_eval_studies",
    ],
)
def test_w1384_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
