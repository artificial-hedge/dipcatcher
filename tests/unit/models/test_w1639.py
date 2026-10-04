import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kitsune_3_qa_studies",
        "tanuki_3_qa_studies",
        "tengu_2_qa_studies",
        "zashiki_warashi_qa_studies",
        "yukionna_qa_studies",
        "dorotabo_qa_studies",
    ],
)
def test_w1639_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
