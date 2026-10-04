import pytest


@pytest.mark.parametrize(
    "name",
    [
        "analogical_prompting_studies",
        "graph_of_thought_studies",
        "least_to_most_studies",
        "plan_and_solve_studies",
        "step_back_studies",
        "tree_of_thought_studies",
    ],
)
def test_w1281_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
