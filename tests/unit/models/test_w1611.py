import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sifaka_qa_studies",
        "fork_marked_qa_studies",
        "fat_tailed_qa_studies",
        "ringtail_qa_studies",
        "needle_clawed_qa_studies",
        "crowned_lemur_qa_studies",
    ],
)
def test_w1611_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
