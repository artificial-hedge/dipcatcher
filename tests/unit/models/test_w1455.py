import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bullfrog_qa_studies",
        "salamander_qa_studies",
        "newt_qa_studies",
        "toad_qa_studies",
        "tree_frog_qa_studies",
        "axolotl_qa_studies",
    ],
)
def test_w1455_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
