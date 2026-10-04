import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lambada_studies",
        "winograd_studies",
        "winogender_studies",
        "wsc_studies",
        "story_cloze_studies",
        "record_studies",
    ],
)
def test_w1307_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
