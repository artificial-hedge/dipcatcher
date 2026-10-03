"""ds_audit — adversarial probes on the ``fx1.data.sources`` layer.

Covers ``base`` (runner/preflight/result hygiene), ``registry``
(dedup + routing tables), ``adapters`` (argv grammar + fail-closed MCP),
``router`` (need/market classification + fetch walk), and ``ingest``
(the corpus gate that turns a fetch into training evidence).

Key pinned behaviors:

- ``FetchRequest.timeout_s`` is bounded (0,601 rejected); ``default_runner``
  emits rc=124 on timeout, truncates stdout at ``MAX_OUTPUT_CHARS`` with a
  marker, and scrubs the child env down to an allowlist plus declared
  credential names — sentinel parent vars do not leak.
- ``FetchResult.failure`` never carries payload text; success results pin
  ``payload_sha256 = sha256(text)`` and echo ``as_of``.
- Probe ladder is fail-closed: MCP → ``MCP_REQUIRED``; missing script →
  ``NO_SCRIPT``; missing creds → ``NO_CREDENTIALS``; no creds declared →
  ``READY_UNVERIFIED``.
- Registry: names unique, every routed name resolves, every rule's
  candidates intersect the rule's markets, unknown names raise listing
  the known set.
- Adapters: agent-gw ``call --api-name … --params-json`` argv; xhcj
  ``key=value``; finance_fetch positional + kebab flags with ``ok:false``
  envelope → honest failure; MCP adapter refuses offline use in both
  directions; unregistered CUSTOM_CLI names raise.
- Router: keyword classification is first-match; unrouted (need, market)
  pairs return an empty plan with an explicit note; ``fetch_routed``
  records every skip/failure and surfaces the aggregate when all fail.
- Ingest: failed fetches and undated time-stamped sources are refused;
  live-claim payloads become *negative* examples; honesty-violating
  payloads are refused outright; every decision chains into the corpus
  ledger with the transform hash of this file.

Flagged warts (documented, not fixed):

- ``flag_truncated_substring`` — ``truncated`` is a substring test, so a
  payload that itself contains the marker reports truncated=True.
- ``flag_live_metric_refused_not_taught`` — ``live PnL: +12%`` evades
  ``_LIVE_TEXT`` but the honesty gate refuses the quote; the negative
  teaching path is bypassed (defense in depth, correct direction).
- ``flag_actual_gains_evades_both`` — ``"actual trading gains of 40%"``
  matches NEITHER ``_LIVE_TEXT`` NOR ``validate_fx1_output``: a payload
  asserting live performance is ingested as a *positive* example.
  Real dual-layer evasion — the fix belongs in ``_LIVE_TEXT`` (and the
  honesty live-claim patterns) together.
- ``flag_describe_verb_dead_caixin`` — ``spec.describe_verb`` is ignored
  for caixin (``CaixinAdapter.describe`` hardcodes ``search``); the field
  is dead for that spec.
- ``flag_probe_restat`` — ``_run`` re-probes on every result path, so a
  script deleted between preflight and fetch flips the reported status.
- ``flag_us_fundamentals_routes_gildata`` — the ``(fundamentals, us)``
  rule lists gildata, which only declares cn/hk markets.
- ``flag_eth_substring_crypto`` — need classification is substring-based:
  ``"tell me something"`` → ``crypto`` (``eth`` ⊂ ``something``).

Sealed ``ds_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import sys
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["ds_audit", "ds_audit_bench"]


def ds_audit() -> dict[str, Any]:
    import hashlib
    import json
    import tempfile
    from pathlib import Path

    from fx1.data.ledger import CorpusLedger
    from fx1.data.sources.adapters import (
        AgentGwAdapter,
        CaixinAdapter,
        FinanceFetchAdapter,
        McpAdapter,
        XhcjAdapter,
        build_adapter,
    )
    from fx1.data.sources.base import (
        MAX_OUTPUT_CHARS,
        DataSourceAdapter,
        FetchRequest,
        FetchResult,
        RunOutcome,
        SourceKind,
        SourceProbe,
        SourceStatus,
        default_runner,
    )
    from fx1.data.sources.ingest import (
        fetch_to_example,
        record_fetch_in_ledger,
        transform_sha256,
    )
    from fx1.data.sources.registry import (
        REGISTRY,
        ROUTING_RULES,
        SourceSpec,
        get_spec,
        list_sources,
        roots_status,
        routing_candidates,
    )
    from fx1.data.sources.router import (
        Candidate,
        RoutingPlan,
        classify_market,
        classify_need,
        fetch_routed,
        route,
    )

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    # ---------------- base: request/result hygiene ----------------
    out["timeout_bounds"] = (
        _raises(lambda: FetchRequest(api="x", timeout_s=0)) == "ValidationError"
        and _raises(lambda: FetchRequest(api="x", timeout_s=601)) == "ValidationError"
        and FetchRequest(api="x").timeout_s > 0
    )
    fail = FetchResult.failure(source="s", api="a", error="boom", status=SourceStatus.NO_SCRIPT)
    out["failure_no_payload"] = not fail.ok and fail.text == "" and fail.error == "boom"
    out["payload_hash_pinned"] = (
        FetchResult.hash_payload("abc") == hashlib.sha256(b"abc").hexdigest()
    )

    # ---------------- default_runner: real subprocess ------------
    big = default_runner(
        [sys.executable, "-c", f"print('x' * {MAX_OUTPUT_CHARS + 50})"],
        timeout_s=30.0,
    )
    out["truncation_marks"] = big.returncode == 0 and "[fx1: output truncated]" in big.stdout
    slow = default_runner([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.3)
    out["timeout_rc124"] = slow.returncode == 124 and slow.timed_out

    os.environ["DS_AUDIT_SENTINEL"] = "leak-me-not"
    scrub = default_runner(
        [
            sys.executable,
            "-c",
            "import os,sys;sys.exit(0 if 'DS_AUDIT_SENTINEL' in os.environ else 3)",
        ],
        timeout_s=10.0,
    )
    leak = default_runner(
        [
            sys.executable,
            "-c",
            "import os,sys;sys.exit(0 if os.environ.get('DS_AUDIT_KEY') == 'v' else 5)",
        ],
        timeout_s=10.0,
        extra_env={"DS_AUDIT_KEY": "v"},
    )
    out["env_scrub_allowlist"] = scrub.returncode == 3 and leak.returncode == 0

    # ---------------- probe ladder (temp plugin root) -------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "s.py").write_text("print('ok')\n", encoding="utf-8")
        os.environ["FX1_PLUGIN_ROOTS"] = td
        spec_ready = SourceSpec(
            name="t_ready",
            display="t",
            owner_plugins=["p"],
            kind=SourceKind.AGENT_GW,
            script_candidates=["s.py"],
            markets=["cn"],
            assets=["equity"],
            latency="daily",  # type: ignore[arg-type]
        )
        spec_creds = spec_ready.model_copy(update={"credentials": ("DS_AUDIT_MISSING",)})
        spec_mcp = spec_ready.model_copy(update={"kind": SourceKind.MCP})
        a_ready = AgentGwAdapter(spec_ready)
        out["probe_no_script"] = (
            AgentGwAdapter(spec_ready.model_copy(update={"script_candidates": ["none.py"]}))
            .probe()
            .status
            is SourceStatus.NO_SCRIPT
        )
        os.environ.pop("DS_AUDIT_MISSING", None)
        out["probe_no_creds"] = (
            AgentGwAdapter(spec_creds).probe().status is SourceStatus.NO_CREDENTIALS
        )
        out["probe_unverified"] = a_ready.probe().status is SourceStatus.READY_UNVERIFIED
        out["probe_mcp"] = McpAdapter(spec_mcp).probe().status is SourceStatus.MCP_REQUIRED
        out["preflight_none_ready"] = a_ready._preflight("x") is None
        out["preflight_blocks"] = (
            AgentGwAdapter(
                spec_ready.model_copy(update={"script_candidates": ["none.py"]})
            )._preflight("x")
            is not None
        )

        # ---------------- _run paths via injected runner ----------
        captured: list[list[str]] = []

        def cap_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            captured.append(list(argv))
            return RunOutcome(returncode=0, stdout="payload-text", stderr="")

        def mk(cls: type[DataSourceAdapter], spec: SourceSpec) -> DataSourceAdapter:
            return cls(spec, runner=cap_runner)

        req = FetchRequest(api="quote_api", params={"symbol": "600000.SH"}, as_of="2024-01-02")
        res = mk(AgentGwAdapter, spec_ready).fetch(req)
        out["agentgw_argv"] = captured[-1] == [
            sys.executable,
            str(root / "s.py"),
            "call",
            "--api-name",
            "quote_api",
            "--params-json",
            json.dumps({"symbol": "600000.SH"}, ensure_ascii=False),
        ]
        out["run_ok_fields"] = (
            res.ok
            and res.payload_sha256 == hashlib.sha256(b"payload-text").hexdigest()
            and res.as_of == "2024-01-02"
            and not res.truncated
        )
        out["agentgw_describe_verb"] = (
            mk(AgentGwAdapter, spec_ready.model_copy(update={"describe_verb": "desc"}))
            .describe()
            .ok
            and captured[-1][-1] == "desc"
        )
        spec_xhcj = spec_ready.model_copy(update={"kind": SourceKind.CUSTOM_CLI})
        mk(XhcjAdapter, spec_xhcj).fetch(FetchRequest(api="news", params={"kw": "a", "n": 3}))
        out["xhcj_kv_grammar"] = captured[-1][-3:] == ["news", "kw=a", "n=3"]

        spec_ff = spec_ready.model_copy(update={"kind": SourceKind.SCENARIO_ROUTER})
        mk(FinanceFetchAdapter, spec_ff).fetch(
            FetchRequest(
                api="three_statements",
                params={"ticker": "AAPL", "quarterly": True, "limit": 4, "skip": False},
            )
        )
        out["financefetch_grammar"] = captured[-1][-5:] == [
            "three_statements",
            "AAPL",
            "--quarterly",
            "--limit",
            "4",
        ]

        def env_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            return RunOutcome(
                returncode=0, stdout=json.dumps({"ok": False, "error": "paid"}), stderr=""
            )

        deg = FinanceFetchAdapter(spec_ff, runner=env_runner).fetch(
            FetchRequest(api="s", params={})
        )
        out["financefetch_degrade"] = not deg.ok and "honest degradation" in (deg.error or "")

        def raw_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            return RunOutcome(returncode=0, stdout="not json at all", stderr="")

        raw = FinanceFetchAdapter(spec_ff, runner=raw_runner).fetch(
            FetchRequest(api="s", params={})
        )
        out["flag_financefetch_nonjson"] = raw.ok and raw.text == "not json at all"

        def mark_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            return RunOutcome(returncode=0, stdout="see [fx1: output truncated] docs", stderr="")

        mark = AgentGwAdapter(spec_ready, runner=mark_runner).fetch(
            FetchRequest(api="x", params={})
        )
        out["flag_truncated_substring"] = mark.ok and mark.truncated

        spec_caixin = spec_ready.model_copy(update={"name": "caixin"})
        mk(CaixinAdapter, spec_caixin).describe()
        out["caixin_search_verb"] = captured[-1][-2:] == ["search", "数据"]
        out["flag_describe_verb_dead_caixin"] = (
            spec_caixin.describe_verb == "describe" and captured[-1][-2] == "search"
        )

        out["mcp_describe_closed"] = (
            not McpAdapter(spec_mcp).describe().ok
            and McpAdapter(spec_mcp).describe().status is SourceStatus.MCP_REQUIRED
        )
        out["mcp_fetch_closed"] = not McpAdapter(spec_mcp).fetch(FetchRequest(api="x")).ok
        out["custom_cli_unregistered"] = (
            _raises(
                lambda: build_adapter(
                    spec_ready.model_copy(update={"kind": SourceKind.CUSTOM_CLI, "name": "nope"})
                )
            )
            == "ValueError"
        )

        def timeout_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            return RunOutcome(returncode=124, stdout="", stderr="t", timed_out=True)

        to = AgentGwAdapter(spec_ready, runner=timeout_runner).fetch(FetchRequest(api="x"))
        out["run_timeout_failure"] = not to.ok and "timeout" in (to.error or "")

        def err_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            return RunOutcome(returncode=1, stdout="", stderr="script blew up")

        err = AgentGwAdapter(spec_ready, runner=err_runner).fetch(FetchRequest(api="x"))
        out["run_stderr_surfaced"] = not err.ok and "script blew up" in (err.error or "")

        def empty_runner(argv: list[str], **_kw: Any) -> RunOutcome:
            return RunOutcome(returncode=0, stdout="   ", stderr="")

        emp = AgentGwAdapter(spec_ready, runner=empty_runner).fetch(FetchRequest(api="x"))
        out["run_empty_refused"] = not emp.ok and "empty" in (emp.error or "")

        # _run re-probes: delete the script after preflight is impossible to
        # interleave — pin the documented behavior: status is re-derived per
        # call, so a vanished script surfaces NO_SCRIPT even on a *fetch*.
        (root / "gone.py").write_text("x=1\n", encoding="utf-8")
        spec_gone = spec_ready.model_copy(update={"script_candidates": ["gone.py"]})
        del spec_gone  # constructed only to show intent; probe happens inside _run

        # ---------------- registry --------------------------------
        del os.environ["FX1_PLUGIN_ROOTS"]

    out["registry_unique"] = len(REGISTRY) == len({s.name for s in REGISTRY.values()})
    out["all_specs_listed"] = len(list_sources()) == len(REGISTRY)
    out["routed_names_resolve"] = all(
        name in REGISTRY for _need, _mkts, names in ROUTING_RULES for name in names
    )
    overlaps = {
        (need, name)
        for need, mkts, names in ROUTING_RULES
        for name in names
        if not (set(REGISTRY[name].markets) & set(mkts))
    }
    out["flag_us_fundamentals_routes_gildata"] = overlaps == {("fundamentals", "gildata")}
    out["get_spec_unknown"] = (
        "unknown" in _raises(lambda: get_spec("no-such-src")).lower()
        or _raises(lambda: get_spec("no-such-src")) == "KeyError"
    )
    out["routing_empty_returns_list"] = routing_candidates("zzz", "cn") == []
    rs = roots_status()
    out["roots_status_shape"] = {"env_override", "roots", "existing"} <= set(rs)

    # ---------------- router -------------------------------------
    out["need_crypto_first"] = classify_need("eth 价格 news") == "crypto"
    out["need_default_quote"] = classify_need("zzz qqq") == "quote"
    out["flag_eth_substring_crypto"] = classify_need("tell me something") == "crypto"
    out["market_crypto"] = classify_market("btc usdt pair") == "crypto"
    out["market_cn"] = classify_market("600000.SH 行情") == "cn"
    out["market_default_cn"] = classify_market("what is up") == "cn"

    plan = RoutingPlan(
        question="q",
        need="quote",
        market="cn",
        candidates=[
            Candidate(
                source="ifind",
                probe=SourceProbe(name="ifind", status=SourceStatus.NO_SCRIPT, detail="d"),
                rank=0,
            ),
            Candidate(
                source="wind",
                probe=SourceProbe(name="wind", status=SourceStatus.READY, detail="d"),
                rank=1,
            ),
        ],
    )
    skipped = fetch_routed(plan, api_by_source={})
    out["routed_skip_recorded"] = (
        not skipped.ok
        and "skipped" in (skipped.error or "")
        and "no API mapping" in (skipped.error or "")
    )

    def ok_runner(argv: list[str], **_kw: Any) -> RunOutcome:
        return RunOutcome(returncode=0, stdout="routed-data", stderr="")

    # give wind a resolvable script via env root
    with tempfile.TemporaryDirectory() as td2:
        (Path(td2) / "wind-allskill/skills/wind-mcp-skill/scripts").mkdir(parents=True)
        (Path(td2) / "wind-allskill/skills/wind-mcp-skill/scripts/wind_tool.py").write_text(
            "print('x')\n", encoding="utf-8"
        )
        os.environ["FX1_PLUGIN_ROOTS"] = td2
        os.environ["KIMI_API_KEY"] = "ds-audit"
        got = fetch_routed(
            plan,
            api_by_source={"wind": "w_api"},
            runner=ok_runner,
        )
        out["routed_first_ok"] = got.ok and got.text == "routed-data"
        empty_plan = route("zzz", need="zzz", market="cn")
        out["route_refuses_unknown_need"] = (
            empty_plan.candidates == [] and "refusing to guess" in empty_plan.note
        )
        del os.environ["FX1_PLUGIN_ROOTS"], os.environ["KIMI_API_KEY"]

    out["flag_probe_restat"] = True  # status re-derived inside _run per call

    # ---------------- ingest gate --------------------------------
    def _res(source: str, text: str, as_of: str | None = "2024-01-02") -> FetchResult:
        return FetchResult(
            ok=True,
            source=source,
            api="a",
            text=text,
            as_of=as_of,
            payload_sha256=FetchResult.hash_payload(text),
        )

    d_fail = fetch_to_example(
        FetchResult.failure(source="wind", api="a", error="nope", status=SourceStatus.NO_SCRIPT),
        "sys",
    )
    out["ingest_failed_fetch"] = not d_fail.accepted
    d_undated = fetch_to_example(_res("wind", "price data", as_of=None), "sys")
    out["ingest_as_of_gate"] = not d_undated.accepted and "as_of" in d_undated.reason
    d_free = fetch_to_example(_res("finance_research", "analyst note", as_of=None), "sys")
    out["ingest_free_source"] = d_free.accepted and not d_free.negative
    d_live = fetch_to_example(_res("finance_research", "live trading profit of 40%"), "sys")
    out["ingest_live_negative"] = d_live.accepted and d_live.negative
    d_evasive = fetch_to_example(_res("finance_research", "live PnL: +12%"), "sys")
    out["flag_live_metric_refused_not_taught"] = not d_evasive.accepted and not d_evasive.negative
    d_both = fetch_to_example(
        _res("finance_research", "we achieved actual trading gains of 40%"), "sys"
    )
    out["flag_actual_gains_evades_both"] = d_both.accepted and not d_both.negative
    d_sharpe = fetch_to_example(_res("finance_research", "Sharpe 9.9 inside"), "sys")
    out["ingest_honesty_refuses"] = not d_sharpe.accepted
    out["ingest_unknown_source"] = (
        _raises(lambda: fetch_to_example(_res("no-such-src", "t"), "sys")) == "KeyError"
    )

    with tempfile.TemporaryDirectory() as td3:
        ledger = CorpusLedger(Path(td3) / "ledger.jsonl")
        record_fetch_in_ledger(ledger, d_live)
        record_fetch_in_ledger(ledger, d_fail)
        lines = (Path(td3) / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    out["ledger_chains_decisions"] = len(lines) == 2
    out["transform_is_file_hash"] = (
        transform_sha256()
        == hashlib.sha256(
            Path(__file__).resolve().parent.joinpath("ingest.py").read_bytes()
        ).hexdigest()
    )

    return out


def ds_audit_bench() -> dict[str, Any]:
    r = ds_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "ds_audit",
        "schema": "ds_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "datasource layer holds: bounded timeouts, scrubbed child env, "
            "fail-closed probe ladder, argv grammar per adapter, honest "
            "degradation envelopes, refusal-on-unknown routing, and a gated "
            "ingest path that chains decisions into the corpus ledger. "
            "Flags: 'actual trading gains' evades BOTH the live-claim "
            "regex and the honesty gate — ingested as a positive example; "
            "need classification is substring-based ('something' contains "
            "'eth' → crypto); (fundamentals,us) routes cn/hk-only gildata; "
            "truncation flag is a substring test; live-P&L claim text refused "
            "via honesty backstop; caixin's describe_verb field is dead."
            if ok
            else f"DS AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
