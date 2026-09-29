# 26 — fx-1 honesty matcher hardening findings

Lane: `src/fx1/honesty.py`, `tests/fx1/test_honesty*.py`. Matching-logic only —
`FORBIDDEN_HEADLINE_TOKENS` is byte-identical before and after (set equality
with `quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS` stays pinned
by `tests/fx1/test_honesty_inheritance.py`).

Closes the §2.2 gaps in `docs/SOTA/15-honesty-extensions.md` that are
closeable inside this lane. The matcher is vendored (ADR-0002 import pin:
`fx1` must not import `quant_fund.leakage.patterns`), following its design.

## 1. What was broken (probed on the pre-change checkout)

`_contains_forbidden_headline` required the number *immediately* after
`\b{token}\b`, on raw text, ASCII-only. All of these slipped (`validate_fx1_output`
returned the text unchanged):

| Evasion | Old result |
|---|---|
| `Sharpe ratio came in at 2.35` | PASS (slips) |
| `sharpe-ratio: 2.35` | PASS (slips) |
| `The Sharpe is 2.4` | PASS (slips) |
| `(Sharpe) = 2.4;` | PASS (slips) |
| `"sharpe": 2.4` | PASS (slips) |
| `P&L was $4,200` / `net asset value peaked at 1.9` | PASS (slips) |
| `2.35 was the Sharpe` | PASS (slips) |
| Cyrillic homoglyph `Shаrpe is 2.1` (U+0430) | PASS (slips) |
| Full-width `Ｓｈａｒｐｅ is 2.1` | PASS (slips) |
| Zero-width `sha\u200brpe is 2.1` | PASS (slips) |
| De-spelling `S h a r p e ratio is 2.35` | PASS (slips) |

## 2. What the hardened matcher does

Pipeline in `fx1.honesty`:

1. **Normalization** (`_normalize_for_match`): NFKC → drop Unicode `Cf`
   category (ZWSP/ZWNJ/ZWJ/word-joiner/BOM/soft-hyphen — these *survive*
   NFKC; verified by probe) → `casefold` → deterministic confusable translate
   (25 Cyrillic/Greek/Latin-lookalike mappings, all ASCII targets, all
   reachable from the mirrored token letters). Full-width and Roman-numeral
   forms are folded by NFKC itself. Idempotent; never applied to returned text.
2. **Alias expansion** (matching logic, *not* new tokens): mirrors
   `leakage.patterns._TOKEN_ALIASES` — `pnl → p&l, p/l, profit and loss`;
   `nav → net asset value`; `sharpe → sharpe ratio`.
3. **Word boundaries** (`_word_source`): alnum/underscore lookarounds (`\b`
   semantics with `_` as a word char, matching the catalog's underscore-token
   key rule). Each alias matches **contiguous OR fully de-spelled**
   (separators `{1,3}` between every character, bounded for linear time), with
   a right boundary that rejects `n a v e l` while allowing `s h a r p e
   ratio is 2.4`.
4. **Claim shape**: token → optional ratio noun (`ratio|value|number|multiple|
   level|score|reading|print`) → bounded connector run (`is/was/of/at/peaked/
   reached/came in at/...`, ≤ 6) → value gap (deliberately excludes `,` `;`
   `.` `_`) → numeric literal (sign, `$€£`, grouped digits, fraction, `%`).
   Plus an inverse pattern for `2.35 was the Sharpe` (number → narrow verb
   set → article → token; no `of/at/in` so counting prose stays clean).
5. **Live-claim patterns** now run on the normalized copy, so a homoglyph /
   full-width `live P&L` claim is caught too.

### False positives explicitly guarded (pinned by tests)

- `navigate`, `navel`, `sharpen`, `nav2` — substring lookalikes.
- `nav_final_dipcatcher 1500764.65`, `init_nav: 1000000.0`, `pnl_total` —
  compound receipt keys (underscore = word char). These land in corpus
  assistant messages from `receipts/`; a naive `\b`-plus-number matcher
  would block the house corpus. Verified by a 241-text sweep over every
  string `validate_fx1_output` touches at build time (eval banks, oracles,
  refusals, DPO pairs, harness stdout, corpus messages): **zero new false
  positives**.
- Counting/enumeration prose: `3 sharpe variants`, `top 5 nav strategies`,
  `sharpe, 3 others`, `P&L; research results are proper scores.`
- Refusals naming the token (`I cannot headline a sharpe figure ...`) — the
  mention-vs-claim contract is preserved.
- ReDoS: connector run capped at 6, all separator quantifiers bounded; a
  5000-separator adversarial string matches in well under a second (test).

### Known matcher asymmetries (accepted, pinned)

- `p and l` spelled with `and` is NOT matched: the leakage-matcher alias table
  has no such entry, and `p and l` appears in ordinary prose
  (`profit and loss`). Both matchers agree.
- Sentence-gap evasion (`Sharpe. Over five panels ... 2.35`) is not matched;
  the connector run is bounded, not a proximity window. This is the
  deliberate precision/recall trade — a proximity window would false-positive
  on the refusal texts and compound-key corpus above.

## 3. Tests added

- `tests/fx1/test_honesty.py`: 110+ cases — boundary lookalikes, compound
  keys, headline phrasings (forward + inverse), unicode evasions (Cyrillic /
  Greek / full-width / zero-width / BOM / soft-hyphen), de-spelling, case /
  whitespace / punctuation invariance, normalization idempotence, ASCII-target
  confusable fold, determinism, linear-time guard, error-message token naming.
- `tests/fx1/test_honesty_inheritance.py`: full bidirectional set equality
  asserted explicitly in *both* directions with named failure messages, plus a
  shape pin (frozenset, non-empty, casefolded, trimmed) so equality cannot
  pass on differently-shaped sets.

## 4. Coverage gaps left open (follow-ups; token-set or cross-lane changes)

1. **Non-English headline phrasing** — `el ratio de Sharpe fue 2.1`,
   `la rentabilidad ... 2.1` still slip. Needs a multilingual connector set
   (or language-detection-then-refuse) — semantics change, cross-lane review.
2. **Semantic synonym renaming** — `risk_adjusted_return_ratio: 2.4` slips
   (documented residual risk, ADR-0008). Closing it means new *tokens*
   (`return_ratio`-style), which is a catalog change (both sets move together).
3. **Proper-score-adjacent concepts not in the banned set**: `information
   ratio`, `alpha`, `beta`, `max drawdown`, `turnover`, `win rate`, `hit
   ratio`, `total return`, `CAGR`, `ulcer index`, `omega`, `kappa`. Rule #1
   bans only Sharpe/Sortino/Calmar/P&L/NAV as headlines; the above are
   commonly headlined in retail contexts. Candidate `FORBIDDEN_RESEARCH_
   METRIC_KEYS` additions — cross-lane (catalog + honesty + inheritance test
   move together).
4. **Claim-local SYNTHETIC labeling** — one `SYNTHETIC` token still launders
   real-data numeric claims elsewhere in the same message (label check is
   whole-text). Fix = proximity-scoped label check; semantics change beyond
   this lane's mandate (risk of blocking the labeled eval-bank prompts, which
   carry one header for many numbers).
5. **Value-side smuggle in structured blobs** — `{"note": "Sharpe 2.1"}` is
   clean for the *key* scan (by design; §2.3 of doc 15). `fx1.honesty` now
   catches it in model *outputs*; artifact-side coverage belongs to
   `build_evidence_report` / verifier lanes.
6. **Spelled-out numbers** — `Sharpe was two point four` slips (no spelled-
   numeral table; `leakage.patterns` has one for its proximity matcher).
7. **Pre-existing unrelated failures on this checkout** (NOT caused by this
   lane; verified by stash-and-rerun): `test_docs_drift.py` (`FX1_TRAINING.md`
   documents `fx1 dpo-build`, CLI ships `fx1 dpo`) and `test_sources.py::
   test_agent_gw_config_file_counts_as_credentials` (sets `HOME`, but Windows
   `Path.home()` reads `USERPROFILE`).
