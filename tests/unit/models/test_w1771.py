import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tarhunna_qa_studies",
        "illuyanka_qa_studies",
        "telepinus_qa_studies",
        "hannahanna_qa_studies",
        "kamrusepa_qa_studies",
        "inara_qa_studies",
    ],
)
def test_w1771_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
