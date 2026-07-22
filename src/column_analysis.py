"""
Figures out which COLUMNS a query actually filters/joins/groups on,
so we can check whether an index covers those specific columns --
not just "does this table have an index on something."

WHY THIS EXISTS: the earlier `has_index` feature (pg_indexes count > 0)
is a real, misleading gap. A table's primary key index makes
`has_index = True` even if the WHERE clause filters on a completely
different, unindexed column -- the exact situation where an index
would actually help and the old signal said "you're fine."

HOW IT WORKS: EXPLAIN's Filter / Index Cond / Hash Cond / Merge Cond /
Join Filter / Recheck Cond fields are raw condition text, e.g.
"(l_shipdate <= '1998-09-02'::date)" or "(o.o_custkey = c.c_custkey)".
Rather than writing a real SQL expression parser (overkill, and
brittle against Postgres's exact formatting), this does word-boundary
matching against each table's REAL column list (from
information_schema.columns) -- if a table's known column name appears
in a condition string, that column was referenced. TPC-H's naming
convention (l_, o_, c_, p_, s_, ps_, n_, r_ prefixes) means column
names rarely collide across tables, so a match is almost always
attributable to exactly one table.
"""

from __future__ import annotations

import re
from typing import Any

_CONDITION_FIELDS = (
    "Filter", "Index Cond", "Hash Cond", "Merge Cond",
    "Join Filter", "Recheck Cond",
)


def _walk_plan(node: dict[str, Any]):
    yield node
    for child in node.get("Plans", []):
        yield from _walk_plan(child)


def _collect_condition_text(plan: dict[str, Any]) -> list[str]:
    """Every raw condition string in the plan, plus Group Key entries
    (which EXPLAIN already gives as a clean list of column names, no
    parsing needed).
    """
    texts: list[str] = []
    for node in _walk_plan(plan):
        for field in _CONDITION_FIELDS:
            if field in node:
                texts.append(str(node[field]))
        texts.extend(node.get("Group Key", []))
        texts.extend(node.get("Sort Key", []))
    return texts


def extract_condition_columns(
    plan: dict[str, Any],
    known_columns: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Returns {table_name: {columns actually referenced in a filter,
    join condition, group by, or sort}} for every table in known_columns.

    known_columns should come from db.get_columns_for_tables() -- the
    table's REAL columns, not a guess -- so matching is exact, not
    pattern-based.
    """
    condition_texts = _collect_condition_text(plan)
    referenced: dict[str, set[str]] = {t: set() for t in known_columns}

    for table, columns in known_columns.items():
        for col in columns:
            pattern = re.compile(r"\b" + re.escape(col) + r"\b")
            if any(pattern.search(text) for text in condition_texts):
                referenced[table].add(col)

    return referenced


def has_relevant_index(
    referenced_columns: dict[str, set[str]],
    indexed_columns: dict[str, set[str]],
) -> bool:
    """True only if EVERY table's referenced columns are covered by
    at least one of that table's indexed columns. A table with no
    referenced columns at all (e.g. a small dimension table scanned
    in full, nothing to filter on) is trivially fine -- there's
    nothing an index could help with.

    Note on composite indexes: this treats any column appearing in
    ANY index as "indexed," including non-leading columns of a
    multi-column index, which Postgres can't always use standalone
    for an equality lookup. Good enough to catch the "no relevant
    index at all" case this was built for; not precise enough to
    distinguish "leading column of a composite index" from "any
    column of it" -- a real refinement if you want to push this further.
    """
    for table, referenced in referenced_columns.items():
        if not referenced:
            continue  # nothing filtered/joined/grouped on this table
        uncovered = referenced - indexed_columns.get(table, set())
        if uncovered:
            return False  # at least one referenced column has zero index coverage
    return True


def missing_index_recommendations(
    referenced_columns: dict[str, set[str]],
    indexed_columns: dict[str, set[str]],
) -> list[str]:
    """Human-readable, per-table, per-column recommendations -- this
    is the piece that resolves the earlier "target column unknown
    from plan alone" limitation in intervention.py. Now it's known.
    """
    recs = []
    for table, referenced in referenced_columns.items():
        uncovered = referenced - indexed_columns.get(table, set())
        if uncovered:
            recs.append(
                f"{table}({', '.join(sorted(uncovered))}) -- referenced in "
                f"a filter/join/group-by but not covered by any index"
            )
    return recs
