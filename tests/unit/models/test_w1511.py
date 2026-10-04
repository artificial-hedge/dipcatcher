import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hermit_qa_studies",
        "woodstar_qa_studies",
        "brilliant_qa_studies",
        "sapphire_qa_studies",
        "topaz_qa_studies",
        "hummingbird_qa_studies",
    ],
)
def test_w1511_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
