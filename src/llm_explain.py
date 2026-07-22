"""
The explanation node -- Phase 2. Narrates a decision that was already
made; does not make decisions itself. See rag.py for why grounding
matters and intervention.py for where the actual decision comes from.

Phase 2 additions: config-driven model/token settings, retries on
transient API failures, and a safe fallback string if the LLM call
fails outright -- a broken explanation should never take down a
pipeline that already successfully executed the query and applied any
intervention.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from classifier import Prediction
from config import LLMConfig
from features import QueryFeatures
from intervention import Branch
from logging_setup import get_logger
import rag

logger = get_logger(__name__)

_PROMPT = ChatPromptTemplate.from_template(
    """You are explaining a query-optimization decision to an engineer.
Be concise (4-6 sentences), specific, and do not invent facts that
aren't in the context below. Where a reference note applies, use it
to make a concrete recommendation, not just a description of the
problem.

Decision: {branch}
Misestimate probability: {probability:.2f}
Table: {table_name}
Estimated rows: {estimated_rows} | Estimated cost: {estimated_cost}
Has index: {has_index} | Seconds since last ANALYZE: {staleness:.0f}
Missing index recommendations: {missing_index_recommendations}
Actions taken: {actions_taken}

Plan shape: top node = {node_type}, {join_count} join(s),
{filtered_seq_scan_count} filtered sequential scan(s),
{index_scan_count} index-backed scan(s)

Query SQL:
{sql}

Relevant reference notes:
{notes}

Explain why this decision was made, in plain language, referencing
the relevant note where it applies. If a reference note suggests a
specific fix (e.g. a partial index, a column order, an autovacuum
setting), name it concretely rather than restating the general
problem."""
)


def _query_terms_for(
    branch: Branch,
    features: QueryFeatures,
    missing_index_recommendations: list[str],
) -> list[str]:
    terms = []
    if features.seconds_since_last_analyze > 3600:
        terms += ["analyze", "stale", "modified"]
        # A table can go a long time between autoanalyze runs simply
        # because it rarely crosses the ~10%-of-rows change threshold,
        # not because autovacuum is unhealthy -- worth distinguishing
        # from a one-off "just run ANALYZE" fix.
        terms += ["autovacuum", "threshold", "churn", "tuning"]
    if not features.has_relevant_index:
        terms += ["index", "scan"]
        # A range-style filter (the common case for numeric/date
        # predicates) often benefits more from a partial index scoped
        # to the condition than a plain whole-column index.
        terms += ["partial", "range", "selective"]
    if features.is_cross_join:
        terms += ["cross", "join", "predicate", "rewrite", "unconditioned"]
    if len(missing_index_recommendations) > 1 or any(
        "," in rec for rec in missing_index_recommendations
    ):
        # More than one missing-index recommendation, or a single
        # recommendation naming multiple columns, means column ORDER
        # in a composite index is relevant, not just presence/absence.
        terms += ["composite", "multicolumn", "order", "leftmost", "prefix"]
    if branch == Branch.WARN_ONLY:
        terms += ["cost", "estimate", "planner"]
    return terms or ["cost", "estimate"]


def _fallback_explanation(branch: Branch, prediction: Prediction) -> str:
    return (
        f"[explanation unavailable -- LLM call failed] "
        f"branch={branch.value}, misestimate_probability={prediction.probability:.2f}"
    )


def build_llm(cfg: LLMConfig) -> ChatOpenAI:

    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        streaming=cfg.streaming,
        top_p=cfg.top_p,
        model_kwargs=cfg.model_kwargs,
    )



def explain_decision(
    branch: Branch,
    prediction: Prediction,
    features: QueryFeatures,
    actions_taken: list[str],
    missing_index_recommendations: list[str],
    sql: str,
    cfg: LLMConfig,
) -> str:
    terms = _query_terms_for(branch, features, missing_index_recommendations)
    # top_k raised from 2 -> 3: the note set grew from 5 to 9 topics,
    # and a single decision can now legitimately touch three distinct
    # concerns at once (e.g. stale stats + missing index + column
    # order) rather than just one or two.
    notes = rag.retrieve(terms, top_k=3)
    notes_text = "\n".join(f"- {n}" for n in notes) if notes else "(none retrieved)"

    llm = build_llm(cfg)

    chain = _PROMPT | llm

    payload = {
        "branch": branch.value,
        "probability": prediction.probability,
        "table_name": features.table_name,
        "estimated_rows": features.estimated_rows,
        "estimated_cost": features.estimated_cost,
        "has_index": features.has_relevant_index,
        "staleness": features.seconds_since_last_analyze,
        "missing_index_recommendations": (
            "; ".join(missing_index_recommendations) if missing_index_recommendations else "none"
        ),
        "actions_taken": ", ".join(actions_taken) if actions_taken else "none",
        "node_type": features.node_type,
        "join_count": features.join_count,
        "filtered_seq_scan_count": features.filtered_seq_scan_count,
        "index_scan_count": features.index_scan_count,
        "sql": sql,
        "notes": notes_text,
    }

    try:
        response = chain.invoke(payload)
        return response.content
    except Exception as exc:  # LLM/network failures shouldn't crash the pipeline
        logger.error("LLM explanation call failed: %s", exc)
        return _fallback_explanation(branch, prediction)
