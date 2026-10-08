"""Web research for the fx-1 front door: fetch, search, and a budgeted,
cancellable deep-research engine with cross-source claim verification.

Every network seam is injectable so the suite runs offline; live runs use
stdlib HTTP only (no API keys, no new dependencies). Reports are labeled
heuristic web verification — never sealed lab receipts.
"""

from fx1.webresearch.engine import (
    Claim,
    ClaimGroup,
    ResearchBudget,
    Researcher,
    ResearchProgress,
    ResearchReport,
    SourceDoc,
    default_queries,
    extract_claims,
    group_claims,
)
from fx1.webresearch.fetch import (
    FetchPolicyError,
    FetchResult,
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
)

__all__ = [
    "Claim",
    "ClaimGroup",
    "DuckDuckGoLiteSearcher",
    "FetchPolicyError",
    "FetchResult",
    "NullSearcher",
    "ResearchBudget",
    "ResearchProgress",
    "ResearchReport",
    "Researcher",
    "SearchHit",
    "SearchResult",
    "Searcher",
    "SourceDoc",
    "default_queries",
    "extract_claims",
    "fetch_page",
    "group_claims",
    "html_to_text",
    "robots_allowed",
    "url_policy_error",
]
