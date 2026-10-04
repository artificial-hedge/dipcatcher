import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tenrec_qa_studies",
        "solenodon_qa_studies",
        "golden_mole_qa_studies",
        "aardvark_qa_studies",
        "elephant_shrew_qa_studies",
        "gymnure_qa_studies",
    ],
)
def test_w1585_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
