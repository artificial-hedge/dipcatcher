import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dill_qa_studies",
        "lemongrass_qa_studies",
        "fennel_qa_studies",
        "mint_qa_studies",
        "nutmeg_qa_studies",
        "clove_qa_studies",
    ],
)
def test_w1473_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
