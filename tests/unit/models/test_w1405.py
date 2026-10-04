import pytest


@pytest.mark.parametrize(
    "name",
    [
        "audio_qa_lite_studies",
        "clotho_qa_studies",
        "avsd_lite_studies",
        "esc_qa_studies",
        "music_avqa_studies",
        "ambi_qa_studies",
    ],
)
def test_w1405_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
