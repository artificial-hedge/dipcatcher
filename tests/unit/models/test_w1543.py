import pytest


@pytest.mark.parametrize(
    "name",
    [
        "poorwill_qa_studies",
        "potoo_qa_studies",
        "pauraque_qa_studies",
        "oilbird_qa_studies",
        "owlet_nightjar_qa_studies",
        "whip_poor_will_qa_studies",
    ],
)
def test_w1543_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
