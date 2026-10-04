import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lechwe_qa_studies",
        "roan_qa_studies",
        "warthog_qa_studies",
        "kob_qa_studies",
        "rhino_qa_studies",
        "buffalo_qa_studies",
    ],
)
def test_w1608_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
