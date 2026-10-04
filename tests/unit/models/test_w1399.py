import pytest


@pytest.mark.parametrize(
    "name",
    [
        "eli5_lite_studies",
        "nq_lite_studies",
        "fresh_qa_studies",
        "trivia_lite_studies",
        "xor_tydi_studies",
        "asqa_lite_studies",
    ],
)
def test_w1399_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
