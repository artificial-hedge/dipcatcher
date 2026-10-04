import pytest


@pytest.mark.parametrize(
    "name",
    [
        "rainbow_serpent_qa_studies",
        "bunyip_qa_studies",
        "yowie_qa_studies",
        "mimis_qa_studies",
        "wandjina_qa_studies",
        "altjira_qa_studies",
    ],
)
def test_w1745_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
