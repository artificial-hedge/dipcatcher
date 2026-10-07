"""Probe: _bench_spectral_atiyah must exercise e2_page (census found it
appended literal True checks without ever calling the module fn)."""

from __future__ import annotations

from quant_fund.models import spectral_atiyah


def test_bench_calls_e2_page(monkeypatch) -> None:
    calls: list[tuple[list[int], list[int]]] = []
    orig = spectral_atiyah.e2_page

    def spy(homology: list[int], coeffs: list[int]) -> list[list[int]]:
        calls.append((homology, coeffs))
        return orig(homology, coeffs)

    monkeypatch.setattr(spectral_atiyah, "e2_page", spy)
    score = spectral_atiyah._bench_spectral_atiyah()
    assert calls, "bench never invoked e2_page — vacuous self-check"
    assert score == 1.0


def test_bench_outputs_positive_and_negative() -> None:
    page = spectral_atiyah.e2_page([-1, 0, -1], [-1, 0, -1])
    # odd homology row and odd coefficient column vanish (negative cases)
    assert page[1] == [0, 0, 0]
    assert page[0][1] == 0
    # Z x Z yields finite-order product entry (positive case)
    assert page[0][0] == 1 and page[2][0] == 1
    out = spectral_atiyah.bench_spectral_atiyah()
    assert out == {"synthetic_spectral_atiyah": 1.0}
    assert all(k.startswith("synthetic_") for k in out)
