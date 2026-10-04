import pytest


@pytest.mark.parametrize(
    "name",
    [
        "argue_eval_studies",
        "mintaka_lite_studies",
        "expert_qa_studies",
        "musique_lite_studies",
        "wiki2_qa_studies",
        "archer_qa_studies",
    ],
)
def test_w1391_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
