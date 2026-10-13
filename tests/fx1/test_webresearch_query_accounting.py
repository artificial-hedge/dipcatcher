"""Executed-query accounting and redirect-budget regressions."""

from collections.abc import Callable

from fx1.webresearch.engine import ResearchBudget, Researcher
from fx1.webresearch.search import SearchResult, Searcher


class _RecordingSearcher(Searcher):
    def __init__(self, on_search: Callable[[str], None] | None = None) -> None:
        self.calls: list[tuple[str, float]] = []
        self.on_search = on_search

    def search(
        self, query: str, *, max_results: int = 8, timeout_s: float = 10.0
    ) -> SearchResult:
        del max_results
        self.calls.append((query, timeout_s))
        if self.on_search is not None:
            self.on_search(query)
        return SearchResult(query=query)


def test_report_lists_only_queries_actually_executed() -> None:
    searcher = _RecordingSearcher()
    report = Researcher(searcher=searcher, now_fn=lambda: 0.0).run(
        "goal",
        queries=["first", "never-run"],
        budget=ResearchBudget(max_searches=1),
    )
    assert report.queries_used == ["first"]


def test_redirect_drops_stale_goal_queries() -> None:
    holder: dict[str, Researcher] = {}

    def redirect_after_old(query: str) -> None:
        if query == "old goal":
            holder["researcher"].redirect("new goal")

    searcher = _RecordingSearcher(on_search=redirect_after_old)
    researcher = Researcher(searcher=searcher, now_fn=lambda: 0.0)
    holder["researcher"] = researcher
    report = researcher.run(
        "old goal",
        queries=["old goal", "stale old-goal query"],
        budget=ResearchBudget(max_searches=2),
    )
    assert [query for query, _ in searcher.calls] == ["old goal", "new goal"]
    assert report.queries_used == ["old goal", "new goal"]
    assert report.redirected_from == ["old goal"]
    assert report.goal == "new goal"


def test_search_receives_remaining_research_deadline() -> None:
    clock = [0.0]

    def advance_after_first(_query: str) -> None:
        if clock[0] == 0.0:
            clock[0] = 8.0

    searcher = _RecordingSearcher(on_search=advance_after_first)
    report = Researcher(searcher=searcher, now_fn=lambda: clock[0]).run(
        "goal",
        queries=["first", "second"],
        budget=ResearchBudget(max_searches=2, max_duration_s=10.0),
    )
    assert searcher.calls == [("first", 10.0), ("second", 2.0)]
    assert report.queries_used == ["first", "second"]
