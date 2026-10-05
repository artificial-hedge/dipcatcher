import pytest


@pytest.mark.parametrize(
    "name",
    [
        "neak_ta_qa_studies",
        "kmoch_qa_studies",
        "arak_qa_studies",
        "mrenh_kongveal_qa_studies",
        "ahp_qa_studies",
        "boramei_qa_studies",
    ],
)
def test_w1941_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
