import pytest


@pytest.mark.parametrize(
    "name",
    [
        "drop_lite_studies",
        "quac_lite_studies",
        "coqa_lite_studies",
        "duo_rc_studies",
        "trivia_web_studies",
        "adver_qa_studies",
    ],
)
def test_w1358_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
