"""
Builds a labeled training set from real queries against your TPC-H
SF10 dataset, trains the classifier, and saves it.

WHY EXPLAIN ANALYZE HERE SPECIFICALLY: this is the one place in the
whole project where EXPLAIN ANALYZE is the right tool. It executes
each query exactly once and returns both the planner's estimate
("Plan Rows") and the ground truth ("Actual Rows") from that single
run -- cheaper than running EXPLAIN then the real query separately.
The online pipeline (graph.py) never does this; it only ever uses
plain EXPLAIN pre-flight, specifically to avoid double execution. This
script is an offline, one-time data-collection exercise, not the live
per-request path -- that distinction is worth being able to explain.

USAGE
    python -m scripts.collect_training_data

This will:
  1. Load the 22 standard queries from tpc-h-queries.sql
  2. Load the 5 supplementary misestimate-stress queries from
     extra_queries.sql
  3. Run EXPLAIN ANALYZE on each (this actually executes every query
     against your live database -- expect this to take a while at
     SF10, since queries like Q9 and Q21 are multi-way joins over
     lineitem's ~60M rows)
  4. Extract features + labels, train, and save the model

CAVEAT WORTH KNOWING BEFORE YOU RUN THIS: 27 queries is a small
training set. The model will fit, but treat its probabilities as
illustrative rather than trustworthy until you've run this against
several different data states (e.g. before/after a manual ANALYZE,
before/after adding an index on a previously-unindexed column) so the
label distribution actually contains both classes in reasonable
numbers -- see the warning collect() logs if it doesn't.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import column_analysis, db, features as feat_mod
from classifier import MisestimateClassifier, compute_label, _vectorize
from config import load_config
from logging_setup import configure_logging, get_logger
from model_evaluation import compare_models, pick_best
from query_loader import load_queries_from_file

# Reuses the precision-floor threshold selection from train_classifier.py
# rather than duplicating it -- see that file for why "lowest threshold
# meeting a precision floor" beats guessing or using a median.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import choose_threshold  # noqa: E402

logger = get_logger(__name__)

_SCRIPT_DIR = Path(Path(__file__).parent).parent
_DEFAULT_QUERY_FILES = [
    str(_SCRIPT_DIR / "sql/tpc-h-queries.sql"),
    str(_SCRIPT_DIR / "sql/extra_queries.sql"),
    str(_SCRIPT_DIR / "sql/optional_expensive_queries.sql"),
]

# _DEFAULT_QUERY_FILES = [
#     str(_SCRIPT_DIR / "sql/tpc-h-queries copy.sql")
# ]
# scripts/optional_expensive_queries.sql is intentionally NOT included
# here -- see that file for why (a real, slow, unfiltered cross join).


def collect(query_files: list[str], schema: str = "ai_ml_experiment"):
    """Runs every query once via EXPLAIN ANALYZE and returns
    (feature_rows, labels, per_query_meta) for training.
    """
    cfg = load_config()
    pool = db.get_pool(cfg.postgres)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {schema}")
        conn.commit()

    feature_rows = []
    labels = []
    meta = []

    all_queries: list[tuple[str, str]] = []
    for path in query_files:
        loaded = load_queries_from_file(path)
        logger.info("loaded %d queries from %s", len(loaded), path)
        all_queries.extend(loaded)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {schema}")
        conn.commit()

        for label, sql_text in all_queries:
            try:
                logger.info("running %s via EXPLAIN ANALYZE...", label)
                plan = db.explain_analyze_query(conn, sql_text)
            except db.QueryExecutionError as exc:
                logger.error("skipping %s -- execution failed: %s", label, exc)
                continue

            relations = feat_mod.extract_relations(plan)
            table_stats = db.get_stats_for_tables(conn, relations)
            table_columns = db.get_columns_for_tables(conn, relations)
            indexed_columns = db.get_indexed_columns_for_tables(conn, relations)

            referenced_columns = column_analysis.extract_condition_columns(plan, table_columns)
            has_relevant_index = column_analysis.has_relevant_index(referenced_columns, indexed_columns)

            query_features = feat_mod.extract_features(plan, table_stats, has_relevant_index)
            actual_rows = float(plan.get("Actual Rows", 0.0))
            row_label = compute_label(query_features.estimated_rows, actual_rows)

            feature_rows.append(query_features)
            labels.append(row_label)
            meta.append({
                "label": label,
                "tables": query_features.tables,
                "estimated_rows": query_features.estimated_rows,
                "actual_rows": actual_rows,
                "misestimated": bool(row_label),
            })
            logger.info(
                "%s: tables=%s estimated_rows=%.0f actual_rows=%.0f misestimated=%s",
                label, query_features.table_name, query_features.estimated_rows,
                actual_rows, bool(row_label),
            )

    return feature_rows, labels, meta





def main():
    cfg = load_config()
    configure_logging(cfg.log_level)

    # query_files = _DEFAULT_QUERY_FILES
    # for path in query_files:
    #     if not os.path.exists(path):
    #         raise SystemExit(f"query file not found: {path}")

    # feature_rows, labels, meta = collect(query_files)

    # import pickle

    # # Bundle everything into one dictionary
    # data_to_save = {"feature_rows": feature_rows, "labels": labels, "meta": meta}

    # # Save as a binary file
    # with open("F:/code_experiment/models/pa_query_data.pkl", "wb") as f:
    #     pickle.dump(data_to_save, f)

    # print("Saved successfully!")

    import pandas as pd

    saved_data = pd.read_pickle("F:/code_experiment/models/query_data.pkl")
    pa_saved_data = pd.read_pickle("F:/code_experiment/models/pa_query_data.pkl")

    # Convert saved_data to DataFrame if it is a dictionary
    if isinstance(saved_data, dict):
        saved_data = pd.DataFrame([saved_data])  # Use [saved_data] if it's a single row, or pd.DataFrame(saved_data) if it's a dict of lists

    # Convert pa_saved_data to DataFrame if it is a dictionary
    if isinstance(pa_saved_data, dict):
        pa_saved_data = pd.DataFrame([pa_saved_data])
     


    # Extract your variables from the saved dict
    feature_rows =  [value for values in saved_data["feature_rows"] + pa_saved_data["feature_rows"] for value in values]
    labels = [value for values in saved_data["labels"] + pa_saved_data["labels"] for value in values]
    meta =  [value for values in saved_data["meta"] + pa_saved_data["meta"] for value in values]

    n_misestimated = sum(labels)
    n_total = len(labels)
    logger.info(
        "collected %d labeled examples (%d misestimated, %d well-estimated)",
        n_total, n_misestimated, n_total - n_misestimated,
    )

    print("\n=== per-query results ===")
    for m in meta:
        print(f"{m['label']:>8}  tables={','.join(m['tables']):<40}  "
              f"est={m['estimated_rows']:>12.0f}  actual={m['actual_rows']:>12.0f}  "
              f"misestimated={m['misestimated']}")

    if n_misestimated < 2 or (n_total - n_misestimated) < 2:
        print(
            f"\nCannot evaluate: only {n_misestimated} misestimated / "
            f"{n_total - n_misestimated} well-estimated examples. Need at "
            "least 2 of each class to cross-validate anything. NOT saving "
            "a model -- there's nothing to evaluate yet. Change data state "
            "(bulk load, drop an index, let stats go stale) and re-run."
        )
        return


   # --- Evaluate BEFORE anything gets saved. This is the fix for "why
    # do we save whatever we get" -- both models are scored the same way
    # via cross-validation, on out-of-fold predictions only, and the
    # comparison is printed regardless of which one ends up chosen. ---
    use_smote = cfg.classifier.use_smote
    if use_smote:
        print(
            "\nSMOTE is enabled (PQE_USE_SMOTE=true). Applied leakage-safe, "
            "inside each CV training fold only -- see model_evaluation.py's "
            "build_estimator() docstring. This does not add real information; "
            "it interpolates between your existing minority-class examples. "
            "Compare the with/without numbers below before trusting the lift."
        )

    print(f"\n=== cross-validated model comparison ({n_total} examples, "
          f"smote={use_smote}) ===")
    results = compare_models(feature_rows, labels, _vectorize, use_smote=use_smote)
    for kind, res in results.items():
        print(f"  {kind:>14}: ROC AUC={res.roc_auc:.3f}  "
              f"avg precision={res.average_precision:.3f}  "
              f"Brier={res.brier_score:.3f}  ({res.n_folds}-fold CV)")

    best_kind = pick_best(results)
    best = results[best_kind]
    print(f"\nSelected model: {best_kind} (highest cross-validated ROC AUC)")

    # A quality gate, stated honestly rather than silently skipped.
    # 0.5 ROC AUC = no better than a coin flip. This does NOT block
    # saving -- at this dataset size, a low score is expected and still
    # worth capturing -- but the model file's metrics are labeled so
    # nobody mistakes an unreliable model for a validated one.
    quality = "unreliable (N too small / AUC near chance)" if best.roc_auc < 0.6 else "provisional"
    if best.roc_auc < 0.6:
        print(
            f"\nWARNING: best ROC AUC is {best.roc_auc:.3f} -- barely better "
            "than chance. This is expected at N={0}, not a bug. Saving anyway "
            "(with metrics labeled 'unreliable') so you have a working "
            "pipeline end-to-end, but do not trust this model's decisions "
            "yet. Re-run after collecting more/varied data.".format(n_total)
        )

    # Threshold from the SAME out-of-fold probabilities used for CV --
    # not from a separate, smaller held-out slice we don't have room for.
    threshold = choose_threshold(labels, best.out_of_fold_probabilities)

    # Final model: refit the chosen kind on ALL data. CV already gave an
    # honest, unbiased performance estimate; there's no remaining reason
    # to withhold any of the 27 rows from the model that actually ships.
    clf = MisestimateClassifier(model_kind=best_kind, use_smote=use_smote)
    clf.fit(feature_rows, labels)
    clf.save(cfg.classifier.model_path)

    metrics_path = str(Path(cfg.classifier.model_path).with_suffix(".metrics.json"))
    metrics_record = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_examples": n_total,
        "n_misestimated": n_misestimated,
        "chosen_model": best_kind,
        "use_smote": use_smote,
        "quality": quality,
        "chosen_high_risk_threshold": round(threshold, 4),
        "model_comparison": {k: v.as_dict() for k, v in results.items()},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"\nModel saved to {cfg.classifier.model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Empirically chosen HIGH_RISK_THRESHOLD: {threshold:.3f}")
    print(f"  export PQE_HIGH_RISK_THRESHOLD={threshold:.3f}")
    print(
        f"\n{n_total} examples is still a small, low-diversity training set. "
        "Rerun collect() after changing data state (bulk loads, "
        "dropped/added indexes, stale vs. fresh ANALYZE) so the classifier "
        "sees genuine variation -- check metrics.json's 'quality' field "
        "before trusting anything this model decides."
    )



if __name__ == "__main__":
    main()


