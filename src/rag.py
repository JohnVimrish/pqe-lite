"""
RAG-lite grounding for the explanation layer.

Deliberately NOT an open web search. The LLM's job is to explain a
decision that's already been made (see intervention.py) -- it doesn't
decide anything itself. Grounding it in a small, fixed, locally-owned
set of reference notes keeps the explanation reproducible and avoids
the LLM inventing plausible-sounding-but-wrong Postgres internals from
memory.

This starter version uses simple keyword overlap instead of real
embeddings, on purpose -- it's enough to prove the pattern end to end.
Swap `retrieve()` for a pgvector similarity search once you want real
semantic retrieval over a larger doc set (fittingly, pgvector is a
Postgres extension, so this can be self-hosted in the same database).

The notes below are original summaries written for this project, not
reproductions of Postgres's official documentation text.
"""

from __future__ import annotations

REFERENCE_NOTES: dict[str, str] = {
    "stale_stats": (
        "Postgres does not update planner statistics automatically on "
        "every write. Statistics only refresh when ANALYZE runs "
        "manually, or when autovacuum's autoanalyze process crosses its "
        "change threshold (roughly 10% of rows plus 50, by default). "
        "Until that happens, the planner keeps estimating row counts "
        "based on the old data distribution, which is a common cause "
        "of cost misestimates after bulk loads or heavy churn."
    ),
    "missing_index": (
        "An index helps the planner avoid a full sequential scan when "
        "a query filters or joins on a condition the index covers. It "
        "does nothing for a query with no filter condition at all, "
        "such as a true CROSS JOIN, since every row must be produced "
        "regardless of any index."
    ),
    "work_mem": (
        "work_mem controls how much memory a single sort or hash "
        "operation can use before spilling intermediate results to "
        "disk. A plan that spills shows this in EXPLAIN ANALYZE output "
        "as an external merge sort or a batched hash join; raising "
        "work_mem for that session can remove the spill, at the cost "
        "of higher peak memory use."
    ),
    "cost_estimate": (
        "EXPLAIN's cost numbers are relative planner cost units, not "
        "wall-clock time predictions. They are derived from table "
        "statistics and configured cost parameters, and are meant to "
        "let the planner compare candidate plans against each other -- "
        "not to promise a specific execution duration."
    ),
    "correlated_columns": (
        "The planner's default statistics assume columns are "
        "statistically independent of each other. When columns are "
        "actually correlated, row estimates can be wrong even with "
        "fresh statistics. CREATE STATISTICS can be used to tell the "
        "planner about specific column correlations it would otherwise "
        "miss."
    ),
    "cross_join_rewrite": (
        "An unconditioned cross join between two subqueries or CTEs "
        "usually indicates a missing join predicate rather than a "
        "genuinely intended full product -- most real queries that "
        "look like a cross join are actually a bug where the join key "
        "was dropped or the WHERE clause only filters one side. Before "
        "accepting the cost estimate as meaningful, verify whether an "
        "intended join key exists between the two sides."
    ),
    "partial_index": (
        "A standard b-tree index covers every row in a column. When a "
        "query repeatedly filters the same column with a narrow, "
        "predictable range or condition (for example, a threshold on a "
        "balance or a specific status value), a partial index scoped "
        "to that condition is smaller, cheaper to maintain, and often "
        "more selective than indexing the whole column."
    ),
    "multicolumn_index_order": (
        "A composite index only helps a query if the query's filter "
        "columns match a left-to-right prefix of the index's column "
        "order -- an index on (a, b, c) helps a filter on a, or on "
        "(a, b), but does not help a filter on b or c alone. The most "
        "selective and most frequently filtered column should "
        "generally be placed first, and column order should match how "
        "the query actually filters, not just which columns are "
        "referenced."
    ),
    "autovacuum_tuning": (
        "Autovacuum's autoanalyze only re-runs once a table's modified "
        "rows cross roughly 10% of its row count plus 50, by default. "
        "A large, rarely-written table can go a very long time between "
        "autoanalyze runs even with autovacuum healthy and running, "
        "because it never crosses that relative threshold. Lowering "
        "the table's autovacuum_analyze_scale_factor, or scheduling a "
        "manual ANALYZE after bulk loads, addresses this more "
        "durably than a one-off ANALYZE."
    ),
}

_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "stale_stats": {"analyze", "stale", "staleness", "autovacuum", "modified"},
    "missing_index": {"index", "cross", "join", "scan"},
    "work_mem": {"work_mem", "spill", "memory", "sort", "hash"},
    "cost_estimate": {"cost", "estimate", "planner"},
    "correlated_columns": {"correlated", "correlation", "misestimate", "statistics"},
    "cross_join_rewrite": {"cross", "join", "predicate", "rewrite", "unconditioned"},
    "partial_index": {"partial", "range", "selective", "threshold"},
    "multicolumn_index_order": {"composite", "multicolumn", "order", "leftmost", "prefix"},
    "autovacuum_tuning": {"autovacuum", "threshold", "churn", "tuning", "scale_factor"},
}


def retrieve(query_terms: list[str], top_k: int = 1) -> list[str]:
    """Keyword-overlap retrieval. Returns the top_k most relevant notes.

    Replace with a real similarity search (e.g. pgvector) once you
    have more than a handful of reference documents -- keyword overlap
    stops scaling once the corpus grows past a few dozen entries.
    """
    query_set = {t.lower() for t in query_terms}
    scored = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        overlap = len(query_set & keywords)
        if overlap > 0:
            scored.append((overlap, topic))
    scored.sort(reverse=True)
    return [REFERENCE_NOTES[topic] for _, topic in scored[:top_k]]
