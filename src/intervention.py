"""
The three-way branch, plus the actions available on "apply_intervention".

Branch logic:

1. A genuine cross join with no filter routes straight to WARN_ONLY
   regardless of probability -- an index can't help a query with
   nothing to look up.
2. Otherwise, the classifier's probability against an empirically
   chosen threshold decides INTERVENE vs. NO_ACTION.
3. High estimated cost with low probability -> WARN_ONLY: the estimate
   is probably correct, there's nothing to fix, but it's worth flagging
   as genuinely expensive.

apply_intervention loops over every table the query touched (real
TPC-H queries join several). Its missing-index flag is now genuinely
actionable: missing_index_recommendations comes from column_analysis.py
via check_stats_node, which matches the SPECIFIC columns this query
filters/joins/groups on against each table's real indexed columns --
not just "does this table have an index somewhere." See
column_analysis.py and features.py for the full reasoning.
"""

from __future__ import annotations

from enum import Enum

from classifier import Prediction
from config import ClassifierConfig
from features import QueryFeatures
from logging_setup import get_logger
import db

logger = get_logger(__name__)


class Branch(str, Enum):
    INTERVENE = "intervene"
    WARN_ONLY = "warn_only"
    NO_ACTION = "no_action"


def decide_branch(
    prediction: Prediction,
    features: QueryFeatures,
    cfg: ClassifierConfig,
) -> Branch:
    if prediction.is_cross_join:
        logger.info(
            "cross join with no filter on [%s] -- routing to warn_only "
            "regardless of probability (%.3f)",
            features.table_name, prediction.probability,
        )
        return Branch.WARN_ONLY

    if prediction.probability >= cfg.high_risk_threshold:
        logger.info(
            "misestimate probability %.3f >= threshold %.3f on [%s] -- intervening",
            prediction.probability, cfg.high_risk_threshold, features.table_name,
        )
        return Branch.INTERVENE

    if features.estimated_cost >= cfg.expensive_cost_threshold:
        logger.info(
            "cost %.0f >= expensive threshold %.0f on [%s], probability %.3f "
            "is low -- warning only, estimate looks correct",
            features.estimated_cost, cfg.expensive_cost_threshold,
            features.table_name, prediction.probability,
        )
        return Branch.WARN_ONLY

    logger.info(
        "probability %.3f, cost %.0f on [%s] -- no action needed",
        prediction.probability, features.estimated_cost, features.table_name,
    )
    return Branch.NO_ACTION


def run_stale_analyze(
    conn,
    features: QueryFeatures,
    table_stats: dict[str, dict],
    cfg: ClassifierConfig,
) -> list[str]:
    """ANALYZE any touched table whose stats are stale, regardless of
    which branch the query ends up on.

    This used to live inside apply_intervention(), which meant it only
    ran on the INTERVENE branch. That's wrong: a query that lands on
    WARN_ONLY (e.g. because it's a genuine cross join, or the estimate
    just happens to be right) can still be sitting on stale stats --
    and the explanation layer will happily tell you stats are stale
    without this ever having a chance to fix it. ANALYZE is cheap and
    always safe to run, so it shouldn't be gated behind the
    misestimate-probability branch decision at all -- see graph.py's
    maintain_stats_node, which now calls this unconditionally right
    after classify_risk, before the three-way branch.
    """
    actions_taken: list[str] = []
    for table in features.tables:
        stats = table_stats.get(table, {})
        staleness = float(stats.get("seconds_since_last_analyze") or 1e9)
        if staleness > cfg.stale_seconds_threshold:
            db.run_analyze(conn, table)
            actions_taken.append(
                f"ran ANALYZE on {table} (stats were {staleness:.0f}s old)"
            )
    return actions_taken


def apply_intervention(
    conn,
    features: QueryFeatures,
    table_stats: dict[str, dict],
    missing_index_recommendations: list[str],
    cfg: ClassifierConfig,
) -> list[str]:
    """Applies cheap, reversible fixes across every table the query touched.

    table_stats and missing_index_recommendations are the same values
    already computed once in check_stats_node -- passed in rather than
    re-derived, since a table's staleness/index coverage shouldn't have
    changed in the few milliseconds since that node ran.

    Stale-stats ANALYZE is now handled separately by run_stale_analyze()
    so it isn't gated behind this being the INTERVENE branch -- this
    function is left with the genuinely INTERVENE-specific action:
    flagging missing indexes for review.
    """
    actions_taken: list[str] = []

    if not features.has_relevant_index and missing_index_recommendations:
        # Genuinely actionable now: names the real table + column(s),
        # not a vague "this table might need an index somewhere."
        for rec in missing_index_recommendations:
            msg = f"flagged missing index: {rec} (not auto-created -- review before adding)"
            logger.warning(msg)
            actions_taken.append(msg)

    if not actions_taken:
        logger.info(
            "intervene branch reached for [%s] but no concrete action applied "
            "-- probability was high without a matching known fix "
            "(worth reviewing as a feature-engineering gap)",
            features.table_name,
        )

    return actions_taken
