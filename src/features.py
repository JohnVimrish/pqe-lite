"""
Turns a raw EXPLAIN plan + per-table stats into a flat feature vector.

INDEX SIGNAL, IMPORTANT DESIGN NOTE: earlier versions of this file used
"does this table have ANY index at all" (via pg_indexes). That's a
real, misleading gap -- a table with an index only on its primary key
still reports "has an index" even when the column actually being
filtered has zero coverage. That's exactly backwards from what the
classifier needs to know.

The fix used here needs no extra DB query at all: the EXPLAIN plan
already tells us, per query, whether an index was actually used for
each table's predicate.
  - A "Seq Scan" node with a "Filter" key means the planner read every
    row and checked the condition row-by-row -- ground truth that no
    usable index existed for THIS predicate, regardless of what other
    indexes the table might have.
  - An "Index Scan" / "Index Only Scan" / "Bitmap Heap Scan" (with an
    "Index Cond"/"Recheck Cond") means an index was actually used.
filtered_seq_scan_count and index_scan_count are built from this
directly. No optimistic default is used anywhere -- if the plan has a
filtered seq scan, that's counted, full stop.

Multi-table aggregation, from the TPC-H rewrite:
- seconds_since_last_analyze: MAX across tables (most-stale table
  drives risk, not the average).
- n_mod_since_analyze / n_live_tup: summed (total churn/size context).
- table_count: more tables in a join means more chances for the
  planner's column-independence assumption to be wrong (see the
  correlated_columns reference note in rag.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from logging_setup import get_logger

logger = get_logger(__name__)

_SCAN_NODE_TYPES = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan"}
_INDEX_BACKED_NODE_TYPES = {"Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Bitmap Index Scan"}


@dataclass
class QueryFeatures:
    node_type: str
    estimated_cost: float
    estimated_rows: float
    plan_width: float
    tables: tuple[str, ...]
    table_count: int
    filtered_seq_scan_count: int
    index_scan_count: int
    has_relevant_index: bool
    seconds_since_last_analyze: float
    n_mod_since_analyze: int
    n_live_tup: int
    is_cross_join: bool
    join_count: int

    @property
    def table_name(self) -> str:
        """Display-friendly label. Real per-table lookups use `tables`."""
        return ", ".join(self.tables) if self.tables else "unknown"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tables"] = ",".join(self.tables)  # flatten for the classifier vector
        return d


def _walk_plan(node: dict[str, Any]):
    """Yield every node in a (possibly nested) EXPLAIN plan tree."""
    yield node
    for child in node.get("Plans", []):
        yield from _walk_plan(child)


def extract_relations(plan: dict[str, Any]) -> list[str]:
    """All base table names actually scanned by this plan, SCHEMA-
    QUALIFIED (e.g. "ai_ml_experiment.customer"), deduplicated, in
    first-seen order. This is what tells us which tables to pull
    staleness stats for -- see check_stats_node in graph.py.

    Schema qualification matters: table names are not guaranteed
    unique across schemas (e.g. a table named `customer` can exist in
    both `public` and a custom schema like `ai_ml_experiment`).
    Without the schema, db.py's stats/column/index lookups filter by
    relname alone, which either matches the wrong schema's table, or
    -- if multiple schemas have a same-named table -- silently
    aggregates rows across schemas that have nothing to do with each
    other (see db.py's docstring on this). EXPLAIN (FORMAT JSON)
    already reports a "Schema" key alongside "Relation Name" on every
    real scan node, so this costs no extra DB round trip.
    """
    seen: list[str] = []
    for node in _walk_plan(plan):
        if node.get("Node Type") in _SCAN_NODE_TYPES:
            rel = node.get("Relation Name")
            if not rel:
                continue
            schema = node.get("Schema") or "ai_ml_experiment"
            qualified = f"{schema}.{rel}"
            if qualified not in seen:
                seen.append(qualified)
    return seen


def _count_joins(plan: dict[str, Any]) -> tuple[int, bool]:
    join_count = 0
    has_cross_join = False
    for node in _walk_plan(plan):
        node_type = node.get("Node Type", "")
        if "Join" in node_type or "Nested Loop" in node_type:
            join_count += 1
            if "Join Filter" not in node and "Hash Cond" not in node \
                    and "Merge Cond" not in node:
                has_cross_join = True
    return join_count, has_cross_join


def find_unindexed_filters(plan: dict[str, Any]) -> list[dict[str, str]]:
    """Every Seq Scan node that filtered rows without an index's help,
    with the table name and the actual filter expression -- used by
    intervention.py to flag exactly which table+predicate is the
    problem, instead of a vague "this table has no index anywhere".
    """
    findings = []
    for node in _walk_plan(plan):
        if node.get("Node Type") == "Seq Scan" and "Filter" in node:
            findings.append({
                "table": node.get("Relation Name", "unknown"),
                "filter": node["Filter"],
            })
    return findings


def _count_scan_types(plan: dict[str, Any]) -> tuple[int, int]:
    filtered_seq_scans = 0
    index_scans = 0
    for node in _walk_plan(plan):
        node_type = node.get("Node Type", "")
        if node_type == "Seq Scan" and "Filter" in node:
            filtered_seq_scans += 1
        elif node_type in _INDEX_BACKED_NODE_TYPES:
            index_scans += 1
    return filtered_seq_scans, index_scans


def extract_features(
    plan: dict[str, Any],
    table_stats: dict[str, dict[str, Any]],
    has_relevant_index: bool,
) -> QueryFeatures:
    """
    table_stats is keyed by table name -- one entry per table returned
    by extract_relations(plan). Tables missing an entry (e.g. a
    brand-new table with no pg_stat_user_tables row yet) are treated
    as maximally stale, the safer default.

    has_relevant_index is computed by the CALLER via column_analysis.py
    + db.get_columns_for_tables()/get_indexed_columns_for_tables()
    (see check_stats_node in graph.py), not here -- that computation
    needs real column names from information_schema, a DB round trip,
    and this function stays a pure transform of an already-fetched
    plan. It's a genuinely different signal from
    filtered_seq_scan_count/index_scan_count below: those reflect what
    the planner actually chose to do at execution time (and can miss
    GROUP BY/ORDER BY entirely, since those don't show up as a scan
    node's Filter); has_relevant_index checks column-level index
    coverage against every column actually referenced in a filter,
    join, group-by, or sort -- independent of what plan the optimizer
    happened to pick this one run. Both are kept as separate features.
    """
    tables = tuple(extract_relations(plan))
    join_count, is_cross_join = _count_joins(plan)
    filtered_seq_scan_count, index_scan_count = _count_scan_types(plan)

    if not tables:
        logger.warning("no base relations found in plan -- unusual for a real query")

    staleness_values = [
        float((table_stats.get(t) or {}).get("seconds_since_last_analyze") or 1e9)
        for t in tables
    ] or [1e9]
    mods_total = sum(int((table_stats.get(t) or {}).get("n_mod_since_analyze") or 0) for t in tables)
    live_tup_total = sum(int((table_stats.get(t) or {}).get("n_live_tup") or 0) for t in tables)

    return QueryFeatures(
        node_type=plan.get("Node Type", "unknown"),
        estimated_cost=float(plan.get("Total Cost", 0.0)),
        estimated_rows=float(plan.get("Plan Rows", 0.0)),
        plan_width=float(plan.get("Plan Width", 0.0)),
        tables=tables,
        table_count=len(tables),
        filtered_seq_scan_count=filtered_seq_scan_count,
        index_scan_count=index_scan_count,
        has_relevant_index=has_relevant_index,
        seconds_since_last_analyze=max(staleness_values),
        n_mod_since_analyze=mods_total,
        n_live_tup=live_tup_total,
        is_cross_join=is_cross_join,
        join_count=join_count,
    )
