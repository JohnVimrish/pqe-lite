"""
Builds a training set from historical (query, actual outcome) pairs
and picks the HIGH_RISK_THRESHOLD empirically via precision/recall --
never by eyeballing a median. See classifier.py and intervention.py
for why.

You need a log of past queries with their real outcomes to run this.
If you don't have one yet, the fastest way to bootstrap it:

  1. Pick a handful of representative queries against your dataset
     (mix of well-indexed, poorly-indexed, joins, cross joins).
  2. For each, capture EXPLAIN (estimate) before running it.
  3. Run it for real, capture actual row count + timing.
  4. Repeat across different data states (before/after bulk loads,
     before/after ANALYZE, before/after adding indexes) so the
     dataset actually contains both misestimated and well-estimated
     examples -- a dataset of only "everything went fine" queries
     can't teach the model what a misestimate looks like.

This script assumes that log already exists as a list of dicts with
keys: plan, table_stats (dict keyed by table name), table_has_index
(dict keyed by table name), actual_rows.

For the concrete, ready-to-run version of this against your actual
TPC-H queries, see scripts/collect_training_data.py -- that script
does the EXPLAIN ANALYZE collection AND calls choose_threshold() from
here. This file's build_training_set()/main() are for the case where
you already have a plain list of historical run dicts from elsewhere
(e.g. a different collection process).
"""

from __future__ import annotations

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from classifier import MisestimateClassifier, compute_label
from config import load_config
from features import extract_features
from logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def build_training_set(historical_runs: list[dict]):
    feature_rows = []
    labels = []
    for run in historical_runs:
        # extract_features() takes a single has_relevant_index bool (the
        # column-level signal from column_analysis.py), not the per-table
        # table_has_index dict this historical-run format stores. Collapse
        # it the same way has_relevant_index is defined elsewhere: only
        # True if every table involved is covered -- one uncovered table
        # is enough to make the query's index coverage incomplete.
        table_has_index = run["table_has_index"]
        has_relevant_index = all(table_has_index.values()) if table_has_index else True

        f = extract_features(
            plan=run["plan"],
            table_stats=run["table_stats"],
            has_relevant_index=has_relevant_index,
        )
        label = compute_label(
            estimated_rows=f.estimated_rows,
            actual_rows=run["actual_rows"],
        )
        feature_rows.append(f)
        labels.append(label)
    return feature_rows, labels


def choose_threshold(y_true, y_proba, min_precision: float = 0.7) -> float:
    """Pick the lowest threshold that still meets a minimum precision bar.

    Framing this as "lowest threshold meeting a precision floor" is a
    deliberate choice: it says "I'd rather catch more real problems
    (higher recall) as long as I don't cry wolf too often (precision
    floor)" -- tune min_precision based on how costly a false
    intervention actually is in your context.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision:
            return float(t)
    return 0.5  # fallback if no threshold meets the bar


def main(historical_runs: list[dict]):
    cfg = load_config()
    configure_logging(cfg.log_level)

    feature_rows, labels = build_training_set(historical_runs)
    if sum(labels) == 0 or sum(labels) == len(labels):
        logger.warning(
            "training set has only one class present (%d misestimated / %d total) "
            "-- you need examples of BOTH misestimated and well-estimated queries, "
            "see this file's module docstring for how to collect them",
            sum(labels), len(labels),
        )

    x_train, x_test, y_train, y_test = train_test_split(
        feature_rows, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = MisestimateClassifier()
    clf.fit(x_train, y_train)

    y_proba = [clf.predict_proba(f).probability for f in x_test]
    threshold = choose_threshold(y_test, y_proba)

    clf.save(cfg.classifier.model_path)
    logger.info("saved trained model to %s", cfg.classifier.model_path)
    print(f"Chosen HIGH_RISK_THRESHOLD: {threshold:.3f}")
    print(f"Set this before running the pipeline:  export PQE_HIGH_RISK_THRESHOLD={threshold:.3f}")
    return clf, threshold


def load_historical_runs_from_mongo() -> list[dict]:
    """NOT YET USABLE FOR RETRAINING AS-IS -- known gap, flagged
    honestly rather than silently wired up wrong.

    The Mongo decision log (mongo_log.py) currently stores a summary
    of each decision (branch, probability, actions taken) -- it does
    NOT store the raw EXPLAIN plan or per-table stats that
    build_training_set() needs to reconstruct features. To make this
    usable, log_outcome_node in graph.py would need to additionally
    store `state["plan"]`, `state["table_stats"]`, and
    `state["table_has_index"]` on every record (they're already
    variable-shape-friendly, so this is a small addition, not a
    redesign) -- just not done yet, since it wasn't needed until this
    function existed.
    """
    import mongo_log

    cfg = load_config()
    collection = mongo_log.get_collection(cfg.mongo)
    return mongo_log.recent_decisions(collection, limit=5000)


if __name__ == "__main__":
    raise SystemExit(
        "Load your historical_runs log and call main(historical_runs) -- "
        "see this file's module docstring for the expected format, or use "
        "load_historical_runs_from_mongo() once you have logged volume."
    )
