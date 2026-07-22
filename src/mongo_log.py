"""
Decision log storage, on MongoDB rather than Postgres.

Why a document store here specifically, and nowhere else in this
project: the three branches (`intervene` / `warn_only` / `no_action`)
produce genuinely different-shaped records. An `intervene` record has
a list of actions taken and staleness numbers; a `no_action` record
has almost nothing beyond the decision itself. Modeling that in
Postgres would mean a wide table full of nullable columns, or a
separate table per branch type with joins just to reconstruct one
run's history. A document per run, shape varying by branch, is the
natural fit -- this is a genuine "the data shape justifies the
database choice" decision, not a keyword.

Every other table-shaped, structured signal in this project (EXPLAIN
plans, stats, features) stays in Postgres deliberately -- see db.py.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from config import MongoConfig
from logging_setup import get_logger

logger = get_logger(__name__)

_client: MongoClient | None = None


def get_collection(cfg: MongoConfig) -> Collection:
    global _client
    if _client is None:
        logger.info("connecting to MongoDB at %s", cfg.uri)
        _client = MongoClient(cfg.uri)
    collection = _client[cfg.db_name][cfg.collection_name]
    # graph.py's log_outcome_node writes a "tables" list per record (a
    # query can touch several tables), not a single "table_name" -- this
    # is a multikey index, so a query matching one table in the list
    # still uses it.
    collection.create_index([("tables", ASCENDING)])
    collection.create_index([("logged_at", ASCENDING)])
    return collection


def log_decision(collection: Collection, record: dict[str, Any]) -> str:
    """Writes one pipeline run's decision record. Returns the inserted _id.

    `record` is expected to already be branch-shaped (see graph.py's
    log_outcome_node) -- this function doesn't enforce a schema on
    purpose, since variable shape is exactly why Mongo was chosen here.
    """
    record = dict(record)
    record.setdefault("logged_at", dt.datetime.utcnow())

    try:
        result = collection.insert_one(record)
    except PyMongoError as exc:
        # A logging failure should never take down the pipeline --
        # the query has already run by the time this executes. Log
        # loudly and move on rather than raising.
        logger.error("failed to write decision log to MongoDB: %s", exc)
        return ""

    return str(result.inserted_id)


def recent_decisions(
    collection: Collection, table_name: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Convenience read path -- e.g. for building a training set later
    from real logged decisions instead of synthetic data.

    table_name matches against the record's "tables" list (a query can
    touch several tables) -- Mongo's implicit array-contains semantics
    mean {"tables": table_name} matches any record whose "tables" list
    includes that value, no $elemMatch/$in needed for a plain equality
    check.
    """
    query = {"tables": table_name} if table_name else {}
    cursor = collection.find(query).sort("logged_at", -1).limit(limit)
    return list(cursor)
