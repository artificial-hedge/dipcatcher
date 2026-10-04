import pytest


@pytest.mark.parametrize(
    "name",
    [
        "woodpigeon_qa_studies",
        "dove_qa_studies",
        "turtle_dove_qa_studies",
        "collared_dove_qa_studies",
        "pigeon_qa_studies",
        "mourning_dove_qa_studies",
    ],
)
def test_w1536_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
