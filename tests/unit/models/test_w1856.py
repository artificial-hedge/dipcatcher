import pytest


@pytest.mark.parametrize(
    "name",
    [
        "esus_qa_studies",
        "taranis_qa_studies",
        "teutates_qa_studies",
        "cernunnos_qa_studies",
        "epona_qa_studies",
        "rosmerta_qa_studies",
    ],
)
def test_w1856_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
