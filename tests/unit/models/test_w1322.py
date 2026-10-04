import pytest


@pytest.mark.parametrize(
    "name",
    [
        "truthfulqa_studies",
        "minif2f_studies",
        "putnam_studies",
        "olympiad_bench_studies",
        "logic_bench_studies",
        "zebra_logic_studies",
    ],
)
def test_w1322_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
