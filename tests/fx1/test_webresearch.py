"""Web research: fetch policy, robots, DDG parsing, engine iteration.

All offline — every network seam is injected. The DDG fixture mirrors the
live lite-endpoint HTML structure (uddg-wrapped result links in one <td>,
snippet in a following <td class='result-snippet'>).
"""

import io
import threading
import urllib.error

import pytest

from fx1.webresearch.engine import (
    Claim,
    ResearchBudget,
    Researcher,
    default_queries,
    extract_claims,
    group_claims,
    tokenize,
)
from fx1.webresearch.fetch import (
    FetchResult,
    clear_robots_cache,
    fetch_page,
    html_to_text,
    robots_allowed,
    url_policy_error,
)
from fx1.webresearch.search import (
    DuckDuckGoLiteSearcher,
    NullSearcher,
    Searcher,
    SearchHit,
    SearchResult,
    _parse_results,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

DDG_LITE_HTML = """
<html><body><table>
  <tr>
    <td>1.</td>
    <td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fone&amp;rut=abc" class='result-link'>First Result Title</a>
    </td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td class='result-snippet'>
      The annual <b>inflation</b> rate was 3.4% in August, according to Labor data.
    </td>
  </tr>
  <tr>
    <td>2.</td>
    <td>
      <a rel="nofollow" href="https://b.example/two" class='result-link'>Second Result</a>
    </td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td class='result-snippet'>Core inflation eased to 2.5% over the same period.</td>
  </tr>
  <tr>
    <td><a rel="nofollow" href="https://ad.example/x" class='result-link'>Sponsored</a></td>
  </tr>
</table></body></html>
"""

PAGE_HTML = """
<html><head><title>Inflation Report</title></head>
<body>
<script>var tracking = "ignored";</script>
<style>.x { color: red }</style>
<h1>US Inflation August 2026</h1>
<p>The annual inflation rate in the United States was 3.4% for the 12 months ending August 2026.</p>
<p>Core inflation rose 2.5% over the year, according to the Labor Department.</p>
<p>See <a href="/related/deep-dive">the deep dive</a> for method details.</p>
<p>Navigate <a href="https://other.example/methodology-inflation">methodology</a>.</p>
</body></html>
"""


@pytest.fixture(autouse=True)
def _clear_robots() -> None:
    clear_robots_cache()


class FakeResponse:
    """Minimal urllib response stand-in (context manager)."""

    def __init__(
        self, body: bytes, status: int = 200, content_type: str = "text/html", url: str = ""
    ) -> None:
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._url = url

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            body, self._body = self._body, b""
            return body
        chunk, self._body = self._body[:n], self._body[n:]
        return chunk

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


# ---------------------------------------------------------------------------
# fetch: html_to_text
# ---------------------------------------------------------------------------


def test_html_to_text_extracts_title_text_links() -> None:
    title, text, links = html_to_text(PAGE_HTML, base_url="https://a.example/report")
    assert title == "Inflation Report"
    assert "3.4% for the 12 months ending August 2026" in text
    assert "var tracking" not in text  # script skipped
    assert "color: red" not in text  # style skipped
    assert "https://a.example/related/deep-dive" in links  # relative resolved
    assert "https://other.example/methodology-inflation" in links
    assert all(link.startswith("http") for link in links)


def test_html_to_text_broken_html_still_yields_text() -> None:
    # <title> is a CDATA element: an unclosed title swallows the document by
    # parser design, so the realistic "broken page" is unclosed body markup.
    title, text, links = html_to_text("<title>T</title><p>hello <b>unclosed", base_url="")
    assert title == "T"
    assert "hello" in text
    assert links == []


# ---------------------------------------------------------------------------
# fetch: URL policy
# ---------------------------------------------------------------------------


def test_url_policy_blocks_non_http_and_private() -> None:
    assert url_policy_error("file:///etc/passwd") is not None
    assert url_policy_error("ftp://example.com/x") is not None
    assert url_policy_error("https://user:pass@example.com/") is not None
    assert url_policy_error("http://127.0.0.1/admin") is not None
    assert url_policy_error("http://192.168.1.1/") is not None
    assert url_policy_error("http://10.0.0.5/") is not None
    assert url_policy_error("http://[::1]/") is not None
    assert url_policy_error("https://example.com/page") is None


def test_url_policy_blocks_dns_rebinding_to_private() -> None:
    def fake_resolve(host: str, port: int) -> list[tuple]:
        return [(2, 1, 6, "", ("10.1.2.3", 80))]

    problem = url_policy_error("http://evil.example/x", resolve_fn=fake_resolve)
    assert problem is not None and "non-public" in problem


def test_url_policy_blocks_unresolvable_host() -> None:
    def fake_resolve(host: str, port: int) -> list[tuple]:
        raise OSError("no such host")

    problem = url_policy_error("http://ghost.invalid/x", resolve_fn=fake_resolve)
    assert problem is not None and "cannot resolve" in problem


def test_fetch_page_refuses_private_without_network() -> None:
    opened: list[str] = []

    def open_fn(request, timeout_s):  # noqa: ANN001
        opened.append(request.full_url)
        raise AssertionError("must never open")

    result = fetch_page("http://127.0.0.1/secret", open_fn=open_fn)
    assert not result.ok
    assert "refused by fetch policy" in result.error
    assert opened == []


# ---------------------------------------------------------------------------
# fetch: fetch_page with injected opener
# ---------------------------------------------------------------------------


def _public_resolve(host: str, port: int) -> list[tuple]:
    return [(2, 1, 6, "", ("93.184.216.34", 80))]


def test_fetch_page_success_parses_html() -> None:
    def open_fn(request, timeout_s):  # noqa: ANN001
        return FakeResponse(PAGE_HTML.encode(), url="https://a.example/report")

    result = fetch_page("https://a.example/report", open_fn=open_fn, resolve_fn=_public_resolve)
    assert result.ok
    assert result.status == 200
    assert result.title == "Inflation Report"
    assert "3.4%" in result.text
    assert result.final_url == "https://a.example/report"
    assert "https://other.example/methodology-inflation" in result.links


def test_fetch_page_caps_body_size() -> None:
    body = b"<html><body>" + b"x" * 1000 + b"</body></html>"

    def open_fn(request, timeout_s):  # noqa: ANN001
        return FakeResponse(body, url="https://a.example/big")

    result = fetch_page(
        "https://a.example/big", open_fn=open_fn, resolve_fn=_public_resolve, max_bytes=100
    )
    assert result.truncated is True
    assert result.size_bytes <= 100


def test_fetch_page_rejects_binary_content_type() -> None:
    def open_fn(request, timeout_s):  # noqa: ANN001
        return FakeResponse(b"\x00\x01", content_type="application/pdf", url="https://a.example/f")

    result = fetch_page("https://a.example/f", open_fn=open_fn, resolve_fn=_public_resolve)
    assert not result.ok
    assert "unsupported content type" in result.error


def test_fetch_page_http_error_reported() -> None:
    def open_fn(request, timeout_s):  # noqa: ANN001
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, io.BytesIO(b""))

    result = fetch_page("https://a.example/x", open_fn=open_fn, resolve_fn=_public_resolve)
    assert not result.ok
    assert result.status == 500
    assert result.error == "HTTP 500"


def test_fetch_page_transport_fault_reported() -> None:
    def open_fn(request, timeout_s):  # noqa: ANN001
        raise urllib.error.URLError("connection refused")

    result = fetch_page("https://a.example/x", open_fn=open_fn, resolve_fn=_public_resolve)
    assert not result.ok
    assert "transport fault" in result.error


# ---------------------------------------------------------------------------
# robots
# ---------------------------------------------------------------------------


def test_robots_allowed_when_robots_missing() -> None:
    def fetch_fn(url: str, **kwargs: object) -> FetchResult:
        return FetchResult(url=url, status=404, error="HTTP 404")

    assert robots_allowed("https://open.example/page", fetch_fn=fetch_fn) is True


def test_robots_disallow_honored() -> None:
    robots = "User-agent: *\nDisallow: /private\n"

    def fetch_fn(url: str, **kwargs: object) -> FetchResult:
        if url.endswith("/robots.txt"):
            return FetchResult(url=url, status=200, text=robots)
        return FetchResult(url=url, status=200, text="page")

    assert robots_allowed("https://site.example/private/data", fetch_fn=fetch_fn) is False
    clear_robots_cache()
    assert robots_allowed("https://site.example/public", fetch_fn=fetch_fn) is True


def test_robots_cached_per_origin() -> None:
    calls: list[str] = []

    def fetch_fn(url: str, **kwargs: object) -> FetchResult:
        calls.append(url)
        return FetchResult(url=url, status=404, error="gone")

    robots_allowed("https://cache.example/a", fetch_fn=fetch_fn)
    robots_allowed("https://cache.example/b", fetch_fn=fetch_fn)
    assert len(calls) == 1  # second call served from cache


# ---------------------------------------------------------------------------
# search: DDG parsing
# ---------------------------------------------------------------------------


def test_parse_results_unwraps_uddg_and_attaches_snippets() -> None:
    hits = _parse_results(DDG_LITE_HTML, "ddg-lite")
    urls = [h.url for h in hits]
    assert "https://a.example/one" in urls  # uddg unwrapped
    assert "https://b.example/two" in urls  # direct link kept
    first = next(h for h in hits if h.url == "https://a.example/one")
    assert first.title == "First Result Title"
    assert "3.4% in August" in first.snippet
    second = next(h for h in hits if h.url == "https://b.example/two")
    assert "Core inflation eased" in second.snippet
    assert all(h.engine == "ddg-lite" for h in hits)


def test_parse_results_dedupes_urls() -> None:
    html = DDG_LITE_HTML.replace("https://b.example/two", "https://a.example/one")
    hits = _parse_results(html, "ddg-lite")
    urls = [h.url for h in hits]
    assert len(urls) == len(set(urls))


def test_searcher_uses_injected_opener() -> None:
    def open_fn(request, timeout_s):  # noqa: ANN001
        assert "q=test" in request.full_url
        return FakeResponse(DDG_LITE_HTML.encode())

    searcher = DuckDuckGoLiteSearcher(open_fn=open_fn)
    result = searcher.search("test query")
    assert result.error == ""
    assert len(result.hits) >= 2


def test_searcher_transport_fault_reports_error_no_hits() -> None:
    def open_fn(request, timeout_s):  # noqa: ANN001
        raise urllib.error.URLError("offline")

    searcher = DuckDuckGoLiteSearcher(open_fn=open_fn)
    result = searcher.search("anything")
    assert result.hits == []
    assert "transport fault" in result.error


def test_null_searcher_is_honest() -> None:
    result = NullSearcher().search("q")
    assert result.hits == []
    assert result.error


# ---------------------------------------------------------------------------
# engine: claims and grouping
# ---------------------------------------------------------------------------


def test_extract_claims_filters_boilerplate_and_short_text() -> None:
    text = (
        "The annual inflation rate was 3.4% in August 2026. "
        "Yes. "
        "The following table contains more values about many things here. "
        "Core inflation rose 2.5% over the same twelve month period."
    )
    claims = extract_claims(text, "US inflation rate")
    texts = [c.text for c in claims]
    assert any("3.4%" in t for t in texts)
    assert not any(t == "Yes." for t in texts)
    assert not any("following table" in t for t in texts)


def test_extract_claims_filters_site_chrome() -> None:
    # Navigation menus say "U.S." too; goal overlap alone must not admit them.
    text = (
        "Tutorial Digital Badges Contact Us My Account Explore Our Apps FRED "
        "Tools and resources to find and use economic data worldwide. "
        "Share this page: Link Copied. "
        "Sign in or subscribe to the newsletter for U.S. data updates. "
        "The annual U.S. inflation rate was 3.4% in August 2026."
    )
    texts = [c.text for c in extract_claims(text, "US inflation rate")]
    assert texts == ["The annual U.S. inflation rate was 3.4% in August 2026."]


def test_extract_claims_respects_max() -> None:
    text = " ".join(f"Sentence number {i} about inflation data here." for i in range(30))
    claims = extract_claims(text, "inflation", max_claims=5)
    assert len(claims) == 5


def test_group_claims_corroborates_across_sources() -> None:
    claims = [
        Claim(
            text="US inflation was 3.4% in August 2026",
            source_url="https://a",
            tokens=frozenset(tokenize("US inflation was 3.4% in August 2026")),
        ),
        Claim(
            text="US inflation was 3.4% in August 2026 per CPI",
            source_url="https://b",
            tokens=frozenset(tokenize("US inflation was 3.4% in August 2026 per CPI")),
        ),
    ]
    groups = group_claims(claims)
    assert len(groups) == 1
    assert groups[0].status == "corroborated"
    assert groups[0].sources == {"https://a", "https://b"}


def test_group_claims_detects_negation_conflict() -> None:
    claims = [
        Claim(
            text="Inflation rose to 3.4% in August 2026",
            source_url="https://a",
            tokens=frozenset(tokenize("Inflation rose to 3.4% in August 2026")),
        ),
        Claim(
            text="Not so: inflation rose to 3.4% in August 2026",
            source_url="https://b",
            tokens=frozenset(tokenize("Not so: inflation rose to 3.4% in August 2026")),
        ),
    ]
    groups = group_claims(claims)
    contested = [g for g in groups if g.status == "contested"]
    assert contested, f"expected a contested group, got {[g.status for g in groups]}"
    assert contested[0].contradictions


def test_group_claims_detects_polarity_conflict() -> None:
    claims = [
        Claim(
            text="Core inflation will increase over the coming year",
            source_url="https://a",
            tokens=frozenset(tokenize("Core inflation will increase over the coming year")),
        ),
        Claim(
            text="Core inflation will decrease over the coming year",
            source_url="https://b",
            tokens=frozenset(tokenize("Core inflation will decrease over the coming year")),
        ),
    ]
    groups = group_claims(claims)
    contested = [g for g in groups if g.status == "contested"]
    assert contested, f"expected a contested group, got {[g.status for g in groups]}"


def test_group_claims_single_source_honest() -> None:
    claims = [
        Claim(
            text="A lonely claim about volatility regimes",
            source_url="https://a",
            tokens=frozenset(tokenize("A lonely claim about volatility regimes")),
        ),
    ]
    groups = group_claims(claims)
    assert groups[0].status == "single-source"


def test_default_queries_include_adversarial_angle() -> None:
    queries = default_queries("check inflation")
    assert queries[0] == "check inflation"
    assert any("criticism" in q or "contradiction" in q for q in queries)


# ---------------------------------------------------------------------------
# engine: iteration, budgets, cancel, redirect
# ---------------------------------------------------------------------------


class FakeSearcher(Searcher):
    """Scripted searcher: query -> hits; records queries; optional hook."""

    def __init__(self, results: dict[str, list[SearchHit]], hook=None) -> None:  # noqa: ANN001
        self.results = results
        self.queries: list[str] = []
        self.hook = hook

    def search(self, query, *, max_results=8, timeout_s=10.0):  # noqa: ANN001
        self.queries.append(query)
        if self.hook is not None:
            self.hook(query)
        hits = self.results.get(query, [])
        return SearchResult(query=query, hits=list(hits[:max_results]))


def make_fetcher(pages: dict[str, tuple[str, str, list[str]]], clock=None):  # noqa: ANN001
    """url -> (title, text, links). robots.txt returns 404 (allowed)."""

    def fetch(url: str, **kwargs: object) -> FetchResult:
        if clock is not None:
            clock.advance()
        if url.endswith("/robots.txt"):
            return FetchResult(url=url, status=404, error="HTTP 404")
        page = pages.get(url)
        if page is None:
            return FetchResult(url=url, status=404, error="HTTP 404")
        title, text, links = page
        return FetchResult(url=url, status=200, title=title, text=text, links=links)

    return fetch


class FakeClock:
    def __init__(self, step: float = 0.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        return self.t

    def advance(self) -> None:
        self.t += self.step


PAGE_A_TEXT = (
    "The annual inflation rate in the United States was 3.4% in August 2026. "
    "Core inflation rose 2.5% over the year according to Labor Department "
    "data released in September 2026."
)
PAGE_B_TEXT = (
    "The annual inflation rate in the United States was 3.4% in August 2026, "
    "according to Bureau of Labor Statistics CPI data. The next inflation "
    "report is scheduled for October 14 at 8:30 in the morning."
)

SMALL_BUDGET = dict(max_duration_s=60, max_depth=0, search_pause_s=0.0)


def test_engine_full_iteration_corroborates_claims() -> None:
    searcher = FakeSearcher(
        {
            "US inflation August 2026": [
                SearchHit(
                    url="https://a.example/rates",
                    title="US inflation rates",
                    snippet="3.4% annual rate",
                ),
                SearchHit(
                    url="https://b.example/cpi", title="CPI August 2026", snippet="inflation 3.4%"
                ),
            ]
        }
    )
    fetcher = make_fetcher(
        {
            "https://a.example/rates": ("Rates", PAGE_A_TEXT, []),
            "https://b.example/cpi": ("CPI", PAGE_B_TEXT, []),
        }
    )
    researcher = Researcher(searcher=searcher, fetch_fn=fetcher, now_fn=FakeClock())
    events: list[str] = []
    report = researcher.run(
        "US inflation August 2026",
        budget=ResearchBudget(max_searches=3, max_sources=4, **SMALL_BUDGET),
        queries=["US inflation August 2026"],
        on_progress=lambda p: events.append(p.event),
    )
    assert report.cancelled is False
    assert len(report.sources) == 2
    assert all(s.ok for s in report.sources)
    assert "started" in events and "searching" in events and "done" in events
    assert "HEURISTIC WEB VERIFICATION" in report.label
    md = report.to_markdown()
    assert "https://a.example/rates" in md
    # both pages carry the same headline sentence -> a multi-source topic
    corroborated = [g for g in report.groups if g.status == "corroborated"]
    assert corroborated, [g.status for g in report.groups]


def test_engine_respects_source_budget() -> None:
    hits = [
        SearchHit(url=f"https://s{i}.example/page", title=f"source {i}", snippet="inflation data")
        for i in range(6)
    ]
    searcher = FakeSearcher({"inflation": hits})
    pages = {
        f"https://s{i}.example/page": (f"S{i}", f"Inflation data point number {i} here.", [])
        for i in range(6)
    }
    researcher = Researcher(searcher=searcher, fetch_fn=make_fetcher(pages), now_fn=FakeClock())
    events: list[str] = []
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(max_searches=5, max_sources=2, **SMALL_BUDGET),
        queries=["inflation"],
        on_progress=lambda p: events.append(p.event),
    )
    assert len(report.sources) == 2
    assert "budget" in events


def test_engine_respects_search_budget() -> None:
    searcher = FakeSearcher({})
    researcher = Researcher(searcher=searcher, fetch_fn=make_fetcher({}), now_fn=FakeClock())
    report = researcher.run(
        "anything",
        budget=ResearchBudget(max_searches=1, max_sources=4, **SMALL_BUDGET),
        queries=["q1", "q2", "q3"],
    )
    assert searcher.queries == ["q1"]
    assert report.sources == []


def test_engine_respects_time_budget() -> None:
    hits = [
        SearchHit(url=f"https://t{i}.example/p", title="inflation", snippet="rate")
        for i in range(5)
    ]
    searcher = FakeSearcher({"inflation": hits})
    pages = {
        f"https://t{i}.example/p": (f"T{i}", f"Inflation rate value {i} recorded here.", [])
        for i in range(5)
    }
    clock = FakeClock(step=10.0)  # each fetch advances 10s past the 15s budget
    researcher = Researcher(
        searcher=searcher, fetch_fn=make_fetcher(pages, clock=clock), now_fn=clock
    )
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(
            max_duration_s=15, max_searches=5, max_sources=5, max_depth=0, search_pause_s=0.0
        ),
        queries=["inflation"],
    )
    assert 1 <= len(report.sources) < 5


def test_engine_cancel_keeps_partial_results() -> None:
    cancel = threading.Event()
    hits = [
        SearchHit(url=f"https://c{i}.example/p", title="inflation", snippet="rate")
        for i in range(4)
    ]
    searcher = FakeSearcher({"inflation": hits})
    pages = {
        f"https://c{i}.example/p": (f"C{i}", f"Inflation rate value {i} here.", [])
        for i in range(4)
    }
    base_fetch = make_fetcher(pages)

    def fetch(url: str, **kwargs: object) -> FetchResult:
        result = base_fetch(url, **kwargs)
        if not url.endswith("/robots.txt"):
            cancel.set()  # operator cancels during the first page fetch
        return result

    researcher = Researcher(searcher=searcher, fetch_fn=fetch, now_fn=FakeClock())
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(max_searches=5, max_sources=4, **SMALL_BUDGET),
        queries=["inflation"],
        cancel_event=cancel,
    )
    assert report.cancelled is True
    assert len(report.sources) == 1  # partial results preserved, loop stopped


def test_engine_redirect_switches_goal() -> None:
    box: dict[str, Researcher] = {}

    def hook(query: str) -> None:
        if query == "old goal":
            box["r"].redirect("new goal")

    searcher = FakeSearcher(
        {
            "old goal": [SearchHit(url="https://old.example/p", title="old", snippet="old goal")],
            "new goal": [SearchHit(url="https://new.example/p", title="new", snippet="new goal")],
        },
        hook=hook,
    )
    pages = {
        "https://old.example/p": ("Old", "Old goal content lives here.", []),
        "https://new.example/p": ("New", "New goal content lives here.", []),
    }
    researcher = Researcher(searcher=searcher, fetch_fn=make_fetcher(pages), now_fn=FakeClock())
    box["r"] = researcher
    events: list[tuple[str, str]] = []
    report = researcher.run(
        "old goal",
        budget=ResearchBudget(max_searches=6, max_sources=6, **SMALL_BUDGET),
        queries=["old goal"],
        on_progress=lambda p: events.append((p.event, p.detail)),
    )
    assert report.goal == "new goal"
    assert report.redirected_from == ["old goal"]
    assert any(e == "redirect" for e, _ in events)
    assert "new goal" in searcher.queries


def test_engine_dedupes_fragment_urls() -> None:
    # In-page anchors are the same document; only one fetch may count.
    hits = [
        SearchHit(url="https://f.example/page", title="inflation", snippet="rate"),
        SearchHit(url="https://f.example/page#section-2", title="inflation", snippet="rate"),
    ]
    searcher = FakeSearcher({"inflation": hits})
    pages = {"https://f.example/page": ("F", "Inflation rate data value here.", [])}
    researcher = Researcher(searcher=searcher, fetch_fn=make_fetcher(pages), now_fn=FakeClock())
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(max_searches=2, max_sources=5, **SMALL_BUDGET),
        queries=["inflation"],
    )
    assert len(report.sources) == 1


def test_engine_dedupes_urls_across_queries() -> None:
    same = SearchHit(url="https://dup.example/p", title="inflation", snippet="rate data")
    searcher = FakeSearcher({"q1": [same], "q2": [same]})
    pages = {"https://dup.example/p": ("Dup", "Inflation rate data value here.", [])}
    researcher = Researcher(searcher=searcher, fetch_fn=make_fetcher(pages), now_fn=FakeClock())
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(max_searches=5, max_sources=5, **SMALL_BUDGET),
        queries=["q1", "q2"],
    )
    assert len(report.sources) == 1


def test_engine_follows_ranked_reference_links() -> None:
    searcher = FakeSearcher(
        {"inflation": [SearchHit(url="https://root.example/p", title="inflation", snippet="rate")]}
    )
    pages = {
        "https://root.example/p": (
            "Root",
            "Inflation rate overview with references to deeper analysis.",
            [
                "https://root.example/inflation-deep-dive",
                "https://root.example/unrelated-cooking",
            ],
        ),
        "https://root.example/inflation-deep-dive": (
            "Deep",
            "Inflation deep dive: the rate was 3.4% in August 2026.",
            [],
        ),
    }
    researcher = Researcher(searcher=searcher, fetch_fn=make_fetcher(pages), now_fn=FakeClock())
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(
            max_searches=2, max_sources=4, max_depth=1, search_pause_s=0.0, max_duration_s=60
        ),
        queries=["inflation"],
    )
    urls = [s.url for s in report.sources]
    assert "https://root.example/inflation-deep-dive" in urls
    assert "https://root.example/unrelated-cooking" not in urls


def test_rank_links_skips_binary_and_scores_path_only() -> None:
    from fx1.webresearch.engine import _rank_links

    links = [
        "https://x.example/inflation-deep-dive",
        "https://x.example/chart-of-inflation.png",
        "https://x.example/report.pdf",
        "https://t.me/share/url?url=https%3A%2F%2Fx.example%2Fp&title=inflation+rate",
    ]
    ranked = _rank_links(links, "inflation")
    assert ranked == ["https://x.example/inflation-deep-dive"]


def test_fetch_page_encodes_non_ascii_urls() -> None:
    seen: list[str] = []

    def open_fn(request, timeout_s):  # noqa: ANN001
        seen.append(request.full_url)
        return FakeResponse(
            b"<html><title>T</title><body>ok text here</body></html>", url=request.full_url
        )

    result = fetch_page(
        "https://a.example/café-inflation—data", open_fn=open_fn, resolve_fn=_public_resolve
    )
    assert result.ok
    assert seen and "caf%C3%A9" in seen[0]
    assert "%E2%80%94" in seen[0]


def test_engine_robots_disallow_skips_source() -> None:
    searcher = FakeSearcher(
        {
            "inflation": [
                SearchHit(url="https://locked.example/p", title="inflation", snippet="rate")
            ]
        }
    )

    def fetch(url: str, **kwargs: object) -> FetchResult:
        if url == "https://locked.example/robots.txt":
            return FetchResult(url=url, status=200, text="User-agent: *\nDisallow: /\n")
        return FetchResult(url=url, status=200, title="P", text="Inflation data here.", links=[])

    researcher = Researcher(searcher=searcher, fetch_fn=fetch, now_fn=FakeClock())
    events: list[tuple[str, str]] = []
    report = researcher.run(
        "inflation",
        budget=ResearchBudget(max_searches=2, max_sources=4, **SMALL_BUDGET),
        queries=["inflation"],
        on_progress=lambda p: events.append((p.event, p.detail)),
    )
    assert report.sources == []
    assert any(e == "skipped" and "robots" in d for e, d in events)


def test_engine_search_errors_recorded_not_fatal() -> None:
    class BrokenSearcher(Searcher):
        def search(self, query, *, max_results=8, timeout_s=10.0):  # noqa: ANN001
            return SearchResult(query=query, error="search transport fault")

    researcher = Researcher(
        searcher=BrokenSearcher(), fetch_fn=make_fetcher({}), now_fn=FakeClock()
    )
    report = researcher.run(
        "x",
        budget=ResearchBudget(max_searches=2, max_sources=2, **SMALL_BUDGET),
        queries=["q"],
    )
    assert report.errors
    assert "transport fault" in report.errors[0]
