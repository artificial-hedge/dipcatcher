import pytest


@pytest.mark.parametrize(
    "name",
    [
        "guivre_qa_studies",
        "tarascon_qa_studies",
        "quinotaur_qa_studies",
        "tarrasque_qa_studies",
        "melusine_qa_studies",
        "gargoyle_qa_studies",
    ],
)
def test_w1657_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
