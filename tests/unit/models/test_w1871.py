import pytest


@pytest.mark.parametrize(
    "name",
    [
        "buggane_qa_studies",
        "fenodyree_qa_studies",
        "tarroo_ushtey_qa_studies",
        "glashtyn_qa_studies",
        "phynnodderee_qa_studies",
        "moddey_dhoo_qa_studies",
    ],
)
def test_w1871_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
