import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nisse_qa_studies",
        "mare_qa_studies",
        "drakk_qa_studies",
        "grimr_qa_studies",
        "sigrun_qa_studies",
        "hildr_qa_studies",
    ],
)
def test_w1671_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
