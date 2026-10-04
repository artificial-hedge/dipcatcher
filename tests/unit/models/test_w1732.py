import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dziewanna_qa_studies",
        "zywie_qa_studies",
        "swarozyc_qa_studies",
        "nija_qa_studies",
        "marzanna_qa_studies",
        "mokosz_qa_studies",
    ],
)
def test_w1732_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
