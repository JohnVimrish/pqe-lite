"""
Database access layer -- psycopg3, pooled connections.

Two design invariants, unchanged from earlier phases:

1. The ONLINE pipeline (explain_query -> ... -> execute_query_timed)
   only ever runs plain EXPLAIN before the real execution. Never
   EXPLAIN ANALYZE there -- that would execute the query twice.

2. Concurrent EXPLAIN/SELECT traffic on the same table is safe under
   MVCC. Only DDL-style operations (CREATE INDEX without CONCURRENTLY,
   ALTER TABLE, VACUUM FULL) take conflicting locks.

Phase 2.1 addition: `explain_analyze_query()`. This is deliberately
NOT used anywhere in the online pipeline (graph.py) -- it exists only
for OFFLINE training-data collection (scripts/collect_training_data.py).
There, a single EXPLAIN ANALYZE run is actually the right and cheapest
tool: it executes the query once and returns both the estimate
("Plan Rows") and the ground truth ("Actual Rows") from that one
execution, instead of running the query twice to get the same
information via two separate calls.

Phase 2.1 addition: multi-table stats/index lookups, since real
TPC-H queries join several tables per query, not one.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import PostgresConfig
from logging_setup import get_logger

logger = get_logger(__name__)


class QueryExecutionError(RuntimeError):
    """Raised when a caller-supplied query fails during EXPLAIN or execution."""


def _split_qualified(name: str, default_schema: str = PostgresConfig.schema) -> tuple[str, str]:
    """Splits a possibly schema-qualified table identifier ('schema.table')
    into (schema, table). Every lookup function below takes the
    SCHEMA-QUALIFIED name that features.extract_relations() now
    produces, and needs both parts: querying pg_stat_user_tables,
    pg_indexes, information_schema.columns, or pg_class by relation
    name alone is ambiguous whenever the same table name exists in
    more than one schema -- it can match the wrong schema's table, or
    (worse) silently return rows unioned across schemas that have
    nothing to do with each other. `default_schema` only applies to a
    plain, unqualified name (e.g. hand-constructed in a test or an
    older stored record) -- normal pipeline usage always supplies a
    qualified name from extract_relations().
    """
    if "." in name:
        schema, _, table = name.partition(".")
        return schema, table
    return default_schema, name


_pool: ConnectionPool | None = None


def get_pool(cfg: PostgresConfig) -> ConnectionPool:
    global _pool
    if _pool is None:
        logger.info("opening Postgres connection pool at %s:%s", cfg.host, cfg.port)
        _pool = ConnectionPool(
            conninfo=cfg.conninfo,
            min_size=cfg.min_pool_size,
            max_size=cfg.max_pool_size,
            open=True,
        )
    return _pool


@contextmanager
def get_connection(pool: ConnectionPool) -> Iterator[psycopg.Connection]:
    with pool.connection() as conn:
        yield conn


def explain_query(conn: psycopg.Connection, sql_text: str) -> dict[str, Any]:
    """Plain EXPLAIN (no ANALYZE). Used by the online pre-flight path."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("EXPLAIN (FORMAT JSON) {}").format(sql.SQL(sql_text)))
            (plan_json,) = cur.fetchone()
    except psycopg.Error as exc:
        logger.error("EXPLAIN failed for query: %s", exc)
        raise QueryExecutionError(f"EXPLAIN failed: {exc}") from exc

    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    return plan_json[0]["Plan"]


def explain_analyze_query(conn: psycopg.Connection, sql_text: str) -> dict[str, Any]:
    """EXPLAIN ANALYZE -- OFFLINE TRAINING USE ONLY. See module docstring.

    Executes `sql_text` for real. Returns the plan tree, where each node
    carries both "Plan Rows" (estimate) and "Actual Rows" (ground truth)
    from this single execution.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("EXPLAIN (ANALYZE, FORMAT JSON) {}").format(sql.SQL(sql_text))
            )
            (plan_json,) = cur.fetchone()
        conn.commit()
    except psycopg.Error as exc:
        conn.rollback()
        logger.error("EXPLAIN ANALYZE failed for training query: %s", exc)
        raise QueryExecutionError(f"EXPLAIN ANALYZE failed: {exc}") from exc

    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    return plan_json[0]["Plan"]


def execute_query_timed(conn: psycopg.Connection, sql_text: str) -> dict[str, Any]:
    """Run the query for real, once, and time it -- the online pipeline's
    only real execution.

    Row DATA is never pulled into Python here -- only elapsed time and
    row count matter downstream (see graph.py's execute_query_node /
    log_outcome_node), so a SELECT gets wrapped in COUNT(*) instead of
    fetchall()'d. This avoids materializing potentially huge result
    sets (e.g. an unfiltered cross join) as Python objects client-side,
    which is what was driving the multi-GB memory blowups -- Postgres
    still does the same work computing the join either way, it just
    never ships the row data over the wire.

    Falls back to the original execute+rowcount path for statements
    where COUNT(*)-wrapping doesn't apply (DML: INSERT/UPDATE/DELETE),
    which raises psycopg.errors.SyntaxError when wrapped this way.
    """
    try:
        with conn.cursor() as cur:
            start = time.perf_counter()
            try:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM ({}) AS pqe_wrapped").format(
                        sql.SQL(sql_text)
                    )
                )
                (row_count,) = cur.fetchone()
            except psycopg.errors.SyntaxError:
                # Not a SELECT-shaped statement (e.g. DML) -- COUNT(*)
                # wrapping doesn't apply. Roll back the failed attempt
                # on this transaction and fall back to direct execution.
                conn.rollback()
                cur.execute(sql.SQL(sql_text))
                try:
                    rows = cur.fetchall()
                    row_count = len(rows)
                except psycopg.ProgrammingError:
                    row_count = cur.rowcount
            elapsed = time.perf_counter() - start
        conn.commit()
    except psycopg.Error as exc:
        conn.rollback()
        logger.error("query execution failed: %s", exc)
        raise QueryExecutionError(f"execution failed: {exc}") from exc

    logger.info("executed query in %.4fs, %d rows", elapsed, row_count)
    return {"elapsed_seconds": elapsed, "row_count": row_count}


_STATS_QUERY = """
    SELECT
        schemaname,
        relname,
        n_live_tup,
        n_mod_since_analyze,
        last_analyze,
        last_autoanalyze,
        EXTRACT(EPOCH from (now() - GREATEST(
            COALESCE(last_analyze, 'epoch'::timestamp),
            COALESCE(last_autoanalyze, 'epoch'::timestamp)
        ))) AS seconds_since_last_analyze
    from pg_stat_user_tables
    WHERE relname = %s AND schemaname = %s
"""


def get_table_stats(conn: psycopg.Connection, qualified_name: str) -> dict[str, Any]:
    schema, table = _split_qualified(qualified_name)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_STATS_QUERY, (table, schema))
        row = cur.fetchone()
    if not row:
        logger.warning(
            "no pg_stat_user_tables entry for %s.%s (new/unused table, or wrong schema?)",
            schema, table,
        )
    return dict(row) if row else {}


def get_stats_for_tables(conn: psycopg.Connection, table_names: list[str]) -> dict[str, dict[str, Any]]:
    """Multi-table version -- one lookup per table touched by a plan."""
    return {t: get_table_stats(conn, t) for t in table_names}


def table_has_index(conn: psycopg.Connection, qualified_name: str) -> bool:
    """"Does this table have ANY index" -- kept for backward compat and
    quick checks, but see get_indexed_columns() below for the real
    signal: an index existing is not the same as an index covering
    the columns a specific query actually filters/joins/groups on.
    """
    schema, table = _split_qualified(qualified_name)
    query = "SELECT COUNT(*) from pg_indexes WHERE tablename = %s AND schemaname = %s"
    with conn.cursor() as cur:
        cur.execute(query, (table, schema))
        (count,) = cur.fetchone()
    return count > 0


def has_index_for_tables(conn: psycopg.Connection, table_names: list[str]) -> dict[str, bool]:
    return {t: table_has_index(conn, t) for t in table_names}


_COLUMNS_QUERY = """
    SELECT column_name from information_schema.columns
    WHERE table_name = %s AND table_schema = %s
"""


def get_table_columns(conn: psycopg.Connection, qualified_name: str) -> set[str]:
    """The table's real column names -- used to disambiguate which
    table a condition string's column references belong to (see
    column_analysis.py).
    """
    schema, table = _split_qualified(qualified_name)
    with conn.cursor() as cur:
        cur.execute(_COLUMNS_QUERY, (table, schema))
        return {row[0] for row in cur.fetchall()}


def get_columns_for_tables(conn: psycopg.Connection, table_names: list[str]) -> dict[str, set[str]]:
    return {t: get_table_columns(conn, t) for t in table_names}


_INDEXED_COLUMNS_QUERY = """
    SELECT a.attname
    from pg_index i
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
    JOIN pg_class c ON c.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relname = %s AND n.nspname = %s
"""


def get_indexed_columns(conn: psycopg.Connection, qualified_name: str) -> set[str]:
    """Every column covered by at least one index on this table --
    the real, column-level signal. Includes non-leading columns of
    composite indexes (see column_analysis.has_relevant_index()'s
    docstring for the nuance that leaves unaddressed).
    """
    schema, table = _split_qualified(qualified_name)
    with conn.cursor() as cur:
        cur.execute(_INDEXED_COLUMNS_QUERY, (table, schema))
        return {row[0] for row in cur.fetchall()}


def get_indexed_columns_for_tables(conn: psycopg.Connection, table_names: list[str]) -> dict[str, set[str]]:
    return {t: get_indexed_columns(conn, t) for t in table_names}


def run_analyze(conn: psycopg.Connection, qualified_name: str) -> None:
    schema, table = _split_qualified(qualified_name)
    logger.info("running ANALYZE on %s.%s", schema, table)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(schema, table)))
        conn.commit()
    except psycopg.Error as exc:
        conn.rollback()
        logger.error("ANALYZE failed on %s.%s: %s", schema, table, exc)
        raise QueryExecutionError(f"ANALYZE failed: {exc}") from exc
