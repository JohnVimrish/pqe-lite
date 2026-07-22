"""
The misestimate classifier.

DESIGN NOTE, still the crux of the whole project: this predicts
P(the planner's cost estimate is wrong), NOT "is this query slow".
A query can be correctly estimated as expensive (a real cross join
over two large tables) -- that's not a misestimate, it's just
expensive, and no fix applies. The three-way decision in
intervention.py depends on keeping this distinction intact.

LABEL DEFINITION (used only at training time, from historical runs):
    ratio = actual_rows / max(estimated_rows, 1)
    misestimated = abs(log(ratio)) > LOG_RATIO_THRESHOLD

A log ratio treats "10x too high" and "10x too low" symmetrically, and
is scale-invariant.

THRESHOLD NOTE: the classifier outputs a probability, not a label.
The cutoff for "high risk" vs "low risk" is chosen empirically via
precision/recall on held-out historical data (see
scripts/train_classifier.py) -- never a hardcoded guess, and never a
median. A median describes typical traffic, not risk.

FEATURE_ORDER matches features.QueryFeatures. Two things worth
knowing about it:

- table_count: more tables in a join means more chances for the
  planner's column-independence assumption to be wrong (see the
  correlated_columns reference note in rag.py).
- filtered_seq_scan_count / index_scan_count REPLACE an earlier,
  weaker "has_index" signal that only checked whether a table had ANY
  index at all -- misleading, since a primary-key-only index made
  that True even when the actual filtered column had zero coverage.
  These two come straight from the EXPLAIN plan's chosen scan type
  instead: a Seq Scan with a Filter means the planner read every row
  and checked the condition row-by-row (no usable index for THIS
  predicate); an Index/Bitmap scan means one was used. Ground truth
  from what the planner actually decided, no extra DB query, no
  optimistic default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from  features import QueryFeatures
from  logging_setup import get_logger

logger = get_logger(__name__)

LOG_RATIO_THRESHOLD = math.log(3.0)  # off by >3x in either direction

FEATURE_ORDER = [
    "estimated_cost",
    "estimated_rows",
    "plan_width",
    "filtered_seq_scan_count",
    "index_scan_count",
    "has_relevant_index",
    "seconds_since_last_analyze",
    "n_mod_since_analyze",
    "n_live_tup",
    "join_count",
    "table_count",
]

# These three routinely span multiple orders of magnitude in real data
# (confirmed directly: seconds_since_last_analyze ranged from 59 to
# 1.78 billion across a merged never-analyzed + freshly-analyzed
# batch; estimated_cost/estimated_rows have the same shape by nature
# of what a query planner estimates). Left raw, StandardScaler's
# mean/std get dominated by the extreme end, which squashes the
# genuinely meaningful small-scale variation down to near nothing --
# degrading these into near-binary signals instead of smooth ones.
# log1p is monotonic, so it doesn't hurt tree-based models (they split
# on relative order, not magnitude) and it's what actually lets
# logistic regression's linear decision boundary use the small-scale
# variation at all.
_LOG_TRANSFORM_FEATURES = {"estimated_cost", "estimated_rows", "seconds_since_last_analyze"}


class ClassifierNotTrainedError(RuntimeError):
    """Raised when predict_proba() is called before fit()/load()."""


def compute_label(estimated_rows: float, actual_rows: float) -> int:
    """1 = misestimated, 0 = estimate was reasonable."""
    ratio = actual_rows / max(estimated_rows, 1.0)
    return int(abs(math.log(max(ratio, 1e-9))) > LOG_RATIO_THRESHOLD)


def _vectorize(f: QueryFeatures) -> list[float]:
    d = f.as_dict()
    values = []
    for name in FEATURE_ORDER:
        v = float(d[name])
        if name in _LOG_TRANSFORM_FEATURES:
            v = math.log1p(v)  # log1p, not log, so a value of 0 doesn't error/−inf
        values.append(v)
    return values


@dataclass
class Prediction:
    probability: float
    is_cross_join: bool  # passed through, not learned -- see intervention.py


class MisestimateClassifier:
    """Thin wrapper so the rest of the pipeline doesn't touch sklearn directly.

    model_kind selects between "logistic" and "random_forest" (see
    model_evaluation.py for why both exist and how they're compared).
    Default stays "logistic" -- it's the safer choice at small sample
    sizes; switch only after model_evaluation.compare_models() shows
    the forest actually winning on your real data, not by default.
    """

    def __init__(self, model_kind: str = "logistic", use_smote: bool = False):
        from  model_evaluation import build_estimator  # local import, avoids a cycle
        self.model_kind = model_kind
        self.use_smote = use_smote
        self._build_estimator = build_estimator
        self.pipeline = None
        self._fitted = False

    def fit(self, feature_rows: list[QueryFeatures], labels: list[int]) -> None:
        if len(feature_rows) < 20:
            logger.warning(
                "fitting the final model on only %d examples -- this call "
                "does NOT evaluate quality. Run model_evaluation.compare_models() "
                "first (collect_training_data.py does this automatically) "
                "before trusting anything this model predicts.",
                len(feature_rows),
            )
        x = np.array([_vectorize(f) for f in feature_rows])
        self.pipeline = self._build_estimator(
            self.model_kind, n_samples=len(feature_rows), use_smote=self.use_smote
        )
        self.pipeline.fit(x, labels)
        self._fitted = True
        logger.info(
            "fitted %s (smote=%s) on %d examples",
            self.model_kind, self.use_smote, len(feature_rows),
        )

    def predict_proba(self, f: QueryFeatures) -> Prediction:
        if not self._fitted:
            raise ClassifierNotTrainedError(
                "Classifier has no training data yet. Run "
                "scripts/collect_training_data.py against your TPC-H queries "
                "first, or load a previously trained model with "
                "MisestimateClassifier.load()."
            )
        x = np.array([_vectorize(f)])
        proba = float(self.pipeline.predict_proba(x)[0][1])
        return Prediction(probability=proba, is_cross_join=f.is_cross_join)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "model_kind": self.model_kind, "use_smote": self.use_smote},
            path,
        )
        logger.info("saved %s classifier to %s", self.model_kind, path)

    @classmethod
    def load(cls, path: str) -> "MisestimateClassifier":
        state = joblib.load(path)
        obj = cls(
            model_kind=state.get("model_kind", "logistic"),
            use_smote=state.get("use_smote", False),
        )
        obj.pipeline = state["pipeline"]
        obj._fitted = True
        logger.info("loaded %s classifier from %s", obj.model_kind, path)
        return obj
