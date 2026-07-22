"""
Cross-validated model evaluation and selection.

Direct response to a real problem: with ~30 labeled queries, a fixed
70/30 train/test split evaluates on ~9 rows, and a three-way
train/val/test split would leave almost nothing per bucket. Three-way
splitting is the right tool once you have hundreds of rows per class,
not thirty total. The right tool here is cross-validation: rotate
which rows are held out across K folds, evaluate on each, average.
You get an honest, lower-variance estimate without permanently
sacrificing a chunk of a tiny dataset.

Also directly fixes: fitting a model and saving it with no evaluation
gate in between. evaluate_model() runs BEFORE anything gets saved, and
returns metrics that get written alongside the model file so
performance is auditable, not just asserted.

Model comparison: logistic regression vs. a calibrated random forest.
Raw RandomForestClassifier.predict_proba() is well known to be poorly
calibrated -- it's "fraction of trees that voted yes," not a real
probability -- which matters a lot here since the whole pipeline's
three-way branch depends on the probability being meaningful, not
just the predicted class. CalibratedClassifierCV fixes that, at the
cost of needing a bit more data to calibrate well. Both models get
evaluated the same way; the data decides which one to use, not
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import QueryFeatures
from logging_setup import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False

MODEL_KINDS = ("logistic", "random_forest") + (("xgboost",) if _HAS_XGBOOST else ())


def build_estimator(kind: str, n_samples: int, use_smote: bool = False):
    """Returns an unfit sklearn-compatible estimator for the given kind.

    n_samples matters here: calibration and tree depth both need to be
    scaled down for small datasets, or they'll overfit/fail outright
    (CalibratedClassifierCV's internal CV can't run with more folds
    than the smaller class has examples).

    use_smote (default False, experimental): wraps the estimator in an
    imblearn Pipeline where SMOTE runs ONLY inside cross_val_predict's
    training fold for each split -- never on the held-out fold. This is
    the leakage-safe way to do it. Applying SMOTE before splitting
    (oversample the whole dataset, then split/CV) lets a synthetic
    point and the real point it was interpolated from land in
    different folds, so the model gets evaluated on a near-duplicate
    of something it already saw -- inflated metrics for the wrong
    reason. imblearn's Pipeline exists specifically to prevent this by
    tying the resampling step to the fold boundary automatically.
    """
    if kind == "logistic":
        base = LogisticRegression(class_weight="balanced")
        steps = [("scaler", StandardScaler())]
    elif kind == "random_forest":
        # Depth and leaf-size deliberately conservative -- at N ~ 30,
        # an unconstrained forest will simply memorize the training
        # rows. These caps matter more than tree count here.
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=max(2, n_samples // 10),
            class_weight="balanced",
            random_state=42,
        )
        calibration_folds = max(2, min(5, n_samples // 10))
        base = CalibratedClassifierCV(rf, method="sigmoid", cv=calibration_folds)
        steps = []
    elif kind == "xgboost":
        if not _HAS_XGBOOST:
            raise ValueError("xgboost is not installed -- pip install xgboost")
        # Same small-N discipline as random_forest: shallow trees, a
        # handful of rounds, heavy L2 regularization. Gradient boosting
        # is, if anything, MORE prone to overfitting tiny data than a
        # random forest, since each tree is explicitly fit to the
        # previous tree's residual errors.
        xgb = XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        calibration_folds = max(2, min(5, n_samples // 10))
        base = CalibratedClassifierCV(xgb, method="sigmoid", cv=calibration_folds)
        steps = []
    else:
        raise ValueError(f"unknown model kind: {kind!r}, expected one of {MODEL_KINDS}")

    if use_smote:
        if not _HAS_IMBLEARN:
            raise ValueError("imbalanced-learn is not installed -- pip install imbalanced-learn")
        # k_neighbors must be less than the smallest class's fold-time
        # count, or SMOTE errors out -- conservative default for small N.
        smote_k = max(1, min(5, n_samples // 10))
        steps = steps + [("smote", SMOTE(k_neighbors=smote_k, random_state=42))]
        steps = steps + [("model", base)]
        return ImbPipeline(steps)

    steps = steps + [("model", base)]
    return Pipeline(steps)


@dataclass
class EvaluationResult:
    model_kind: str
    n_samples: int
    n_folds: int
    roc_auc: float
    average_precision: float
    brier_score: float
    out_of_fold_probabilities: list[float]

    def as_dict(self) -> dict:
        return {
            "model_kind": self.model_kind,
            "n_samples": self.n_samples,
            "n_folds": self.n_folds,
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "brier_score": round(self.brier_score, 4),
        }


def evaluate_model(
    feature_rows: list[QueryFeatures],
    labels: list[int],
    model_kind: str,
    vectorize_fn,
    use_smote: bool = False,
) -> EvaluationResult:
    """Cross-validated evaluation. Returns metrics computed ONLY from
    out-of-fold predictions -- no row is ever scored by a model that
    saw it during training, so this can't quietly overstate quality.

    use_smote: experimental, off by default -- see build_estimator()'s
    docstring for why this is only safe when wired through
    cross_val_predict this way (SMOTE re-runs fresh inside each
    training fold, never touching that fold's held-out rows).
    """
    x = np.array([vectorize_fn(f) for f in feature_rows])
    y = np.array(labels)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    min_class_count = min(n_pos, n_neg) if n_pos and n_neg else 0

    if min_class_count < 2:
        raise ValueError(
            f"cannot cross-validate: only {n_pos} misestimated / {n_neg} "
            "well-estimated examples. Need at least 2 of each class. "
            "Re-run collection after changing data state (see "
            "collect_training_data.py's module docstring)."
        )

    n_folds = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    estimator = build_estimator(model_kind, n_samples=len(y), use_smote=use_smote)
    oof_proba = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]

    result = EvaluationResult(
        model_kind=model_kind,
        n_samples=len(y),
        n_folds=n_folds,
        roc_auc=float(roc_auc_score(y, oof_proba)),
        average_precision=float(average_precision_score(y, oof_proba)),
        brier_score=float(brier_score_loss(y, oof_proba)),
        out_of_fold_probabilities=oof_proba.tolist(),
    )
    logger.info(
        "%s: %d-fold CV -- ROC AUC %.3f, avg precision %.3f, Brier %.3f",
        model_kind, n_folds, result.roc_auc, result.average_precision, result.brier_score,
    )
    return result


def compare_models(
    feature_rows: list[QueryFeatures], labels: list[int], vectorize_fn,
    use_smote: bool = False,
) -> dict[str, EvaluationResult]:
    """Evaluates every model kind the same way and returns all results,
    so the choice is made from evidence, not assumption.
    """
    results = {}
    for kind in MODEL_KINDS:
        try:
            results[kind] = evaluate_model(
                feature_rows, labels, kind, vectorize_fn, use_smote=use_smote
            )
        except ValueError as exc:
            logger.warning("skipping %s: %s", kind, exc)
    return results


def pick_best(results: dict[str, EvaluationResult]) -> str:
    """ROC AUC as the primary selection criterion -- it's threshold-
    independent, which matters since the actual operating threshold
    gets chosen separately (see train_classifier.choose_threshold).
    Ties broken toward logistic regression: at small N, the simpler,
    more stable model is the safer default even at equal AUC.
    """
    if not results:
        raise ValueError("no model evaluated successfully -- nothing to pick from")
    best_kind = max(
        results,
        key=lambda k: (round(results[k].roc_auc, 3), k == "logistic"),
    )
    return best_kind