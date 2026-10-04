import pytest


@pytest.mark.parametrize(
    "name",
    [
        "salamander_2_qa_studies",
        "undine_qa_studies",
        "gnome_2_qa_studies",
        "sylph_2_qa_studies",
        "ifrit_qa_studies",
        "marid_qa_studies",
    ],
)
def test_w1634_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
