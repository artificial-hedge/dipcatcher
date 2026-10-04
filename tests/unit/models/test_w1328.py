import pytest


@pytest.mark.parametrize(
    "name",
    [
        "polyglot_bench_studies",
        "multipl_e_studies",
        "apps_bench_studies",
        "code_contests_studies",
        "repobench_studies",
        "class_eval_studies",
    ],
)
def test_w1328_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
