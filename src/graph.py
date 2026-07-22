"""
LangGraph wiring for the full pipeline. Node names match the boxes in
the architecture diagram 1:1.

Why LangGraph here: classify_risk branches to one of three paths that
all reconverge before execute_query -- a plain linear LangChain chain
doesn't model that branch+fan-in well.

Index signal: check_stats_node computes has_relevant_index via
column_analysis.py -- column-level index coverage against the
columns THIS query actually filters/joins/groups on, not just
"does this table have some index somewhere." See column_analysis.py
and features.py's module docstrings for why that distinction matters.
"""

from __future__ import annotations

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from psycopg_pool import ConnectionPool
from pymongo.collection import Collection

import column_analysis, db, features as feat_mod, llm_explain, mongo_log
from classifier import MisestimateClassifier, Prediction
from config import AppConfig
from features import QueryFeatures
from intervention import Branch, decide_branch, apply_intervention, run_stale_analyze
from logging_setup import get_logger

logger = get_logger(__name__)


class PipelineState(TypedDict, total=False):
    sql: str
    pg_pool: ConnectionPool
    mongo_collection: Collection
    classifier: MisestimateClassifier
    config: AppConfig
    plan: dict
    tables: list[str]
    table_stats: dict[str, dict]
    table_columns: dict[str, set]
    indexed_columns: dict[str, set]
    referenced_columns: dict[str, set]
    has_relevant_index: bool
    missing_index_recommendations: list[str]
    query_features: QueryFeatures
    prediction: Prediction
    branch: Branch
    maintenance_actions: list[str]
    actions_taken: list[str]
    execution_result: dict
    explanation: str
    error: Optional[str]


def explain_preflight_node(state: PipelineState) -> PipelineState:
    with state["pg_pool"].connection() as conn:
        plan = db.explain_query(conn, state["sql"])
    return {"plan": plan}


def check_stats_node(state: PipelineState) -> PipelineState:
    tables = feat_mod.extract_relations(state["plan"])
    with state["pg_pool"].connection() as conn:
        table_stats = db.get_stats_for_tables(conn, tables)
        table_columns = db.get_columns_for_tables(conn, tables)
        indexed_columns = db.get_indexed_columns_for_tables(conn, tables)

    # Which columns does THIS query actually filter/join/group on, per
    # table -- and does an index cover them? Column-level, not the
    # weaker "table has some index" check. See column_analysis.py.
    referenced_columns = column_analysis.extract_condition_columns(state["plan"], table_columns)
    has_relevant_index = column_analysis.has_relevant_index(referenced_columns, indexed_columns)
    missing_index_recommendations = column_analysis.missing_index_recommendations(
        referenced_columns, indexed_columns
    )

    return {
        "tables": tables,
        "table_stats": table_stats,
        "table_columns": table_columns,
        "indexed_columns": indexed_columns,
        "referenced_columns": referenced_columns,
        "has_relevant_index": has_relevant_index,
        "missing_index_recommendations": missing_index_recommendations,
    }


def classify_risk_node(state: PipelineState) -> PipelineState:
    query_features = feat_mod.extract_features(
        plan=state["plan"],
        table_stats=state["table_stats"],
        has_relevant_index=state["has_relevant_index"],
    )
    prediction = state["classifier"].predict_proba(query_features)
    branch = decide_branch(prediction, query_features, state["config"].classifier)
    logger.info("misestimate probability=%.3f branch=%s", prediction.probability, branch.value)
    return {
        "query_features": query_features,
        "prediction": prediction,
        "branch": branch,
    }


def maintain_stats_node(state: PipelineState) -> PipelineState:
    """Runs ANALYZE on any touched table with stale stats, unconditionally
    -- before the three-way branch, not gated behind it. See
    intervention.run_stale_analyze()'s docstring for why this used to
    be (wrongly) INTERVENE-only.
    """
    with state["pg_pool"].connection() as conn:
        actions = run_stale_analyze(
            conn, state["query_features"], state["table_stats"], state["config"].classifier,
        )
    return {"maintenance_actions": actions}


def apply_intervention_node(state: PipelineState) -> PipelineState:
    with state["pg_pool"].connection() as conn:
        actions = apply_intervention(
            conn, state["query_features"], state["table_stats"],
            state["missing_index_recommendations"], state["config"].classifier,
        )
    logger.info("applying intervention: %s", actions)
    return {"actions_taken": state.get("maintenance_actions", []) + actions}


def warn_only_node(state: PipelineState) -> PipelineState:
    return {
        "actions_taken": state.get("maintenance_actions", [])
        + ["none -- cost estimate appears correct, warning only"]
    }


def no_action_node(state: PipelineState) -> PipelineState:
    return {"actions_taken": state.get("maintenance_actions", [])}


def execute_query_node(state: PipelineState) -> PipelineState:
    with state["pg_pool"].connection() as conn:
        result = db.execute_query_timed(conn, state["sql"])
    return {"execution_result": result}


def explain_decision_node(state: PipelineState) -> PipelineState:
    explanation = llm_explain.explain_decision(
        branch=state["branch"],
        prediction=state["prediction"],
        features=state["query_features"],
        actions_taken=state["actions_taken"],
        missing_index_recommendations=state["missing_index_recommendations"],
        sql=state["sql"],
        cfg=state["config"].llm,
    )
    return {"explanation": explanation}


def log_outcome_node(state: PipelineState) -> PipelineState:
    """Builds the branch-shaped decision record and writes it to Mongo.
    Record shape genuinely varies by branch -- see mongo_log.py.
    """
    record = {
        "tables": list(state["query_features"].tables),
        "sql": state["sql"],
        "branch": state["branch"].value,
        "misestimate_probability": state["prediction"].probability,
        "execution": state["execution_result"],
        "explanation": state["explanation"],
    }

    if state["branch"] == Branch.INTERVENE:
        record["actions_taken"] = state["actions_taken"]
        record["staleness_seconds"] = state["query_features"].seconds_since_last_analyze
        record["missing_index_recommendations"] = state["missing_index_recommendations"]
    elif state["branch"] == Branch.WARN_ONLY:
        record["estimated_cost"] = state["query_features"].estimated_cost
        record["is_cross_join"] = state["query_features"].is_cross_join

    inserted_id = mongo_log.log_decision(state["mongo_collection"], record)
    logger.info(
        "logged decision %s for [%s] (_id=%s)",
        state["branch"].value, state["query_features"].table_name, inserted_id,
    )
    return {}


def _route_after_classify(state: PipelineState) -> str:
    return state["branch"].value


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("explain_preflight", explain_preflight_node)
    graph.add_node("check_stats", check_stats_node)
    graph.add_node("classify_risk", classify_risk_node)
    graph.add_node("maintain_stats", maintain_stats_node)
    graph.add_node("apply_intervention", apply_intervention_node)
    graph.add_node("warn_only", warn_only_node)
    graph.add_node("no_action", no_action_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("explain_decision", explain_decision_node)
    graph.add_node("log_outcome", log_outcome_node)

    graph.set_entry_point("explain_preflight")
    graph.add_edge("explain_preflight", "check_stats")
    graph.add_edge("check_stats", "classify_risk")
    graph.add_edge("classify_risk", "maintain_stats")

    graph.add_conditional_edges(
        "maintain_stats",
        _route_after_classify,
        {
            Branch.INTERVENE.value: "apply_intervention",
            Branch.WARN_ONLY.value: "warn_only",
            Branch.NO_ACTION.value: "no_action",
        },
    )

    graph.add_edge("apply_intervention", "execute_query")
    graph.add_edge("warn_only", "execute_query")
    graph.add_edge("no_action", "execute_query")

    graph.add_edge("execute_query", "explain_decision")
    graph.add_edge("explain_decision", "log_outcome")
    graph.add_edge("log_outcome", END)

    return graph.compile()
