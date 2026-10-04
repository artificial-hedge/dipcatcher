import pytest


@pytest.mark.parametrize(
    "name",
    [
        "math_reason_studies",
        "arith_qa_studies",
        "gsm_hard_studies",
        "theorem_qa_studies",
        "proof_pile_studies",
        "mini_f2f_studies",
    ],
)
def test_w1338_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
