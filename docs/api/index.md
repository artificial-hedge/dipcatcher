# API reference

Generated from the docstrings in `src/quant_fund`. The pages cover the public research surface: the data lake, Northset microstructure helpers, validation gates, proper scores, and receipt verification.

Import paths match the package. Build this site from a checkout with `make docs` (`uv run --only-group docs`). The handler loads the library from `src/` and renders signatures plus docstrings. Undocumented private helpers (names starting with `_`) are omitted.

These modules do not submit orders. The armed order gateway is documented in [Live order gateway](../LIVE_TRADING.md).

- [quant_fund.data](data.md)
- [quant_fund.microstructure](microstructure.md)
- [quant_fund.validation](validation.md)
- [quant_fund.metrics](metrics.md)
- [quant_fund.research.verify](research_verify.md)
