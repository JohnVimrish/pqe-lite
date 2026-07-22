# PQE-lite

A predictive plan-correction layer for Postgres, built as a portfolio
project bridging ML-on-infrastructure-telemetry, systems/database
internals, and LLM-based reasoning — three skills that rarely show up
together in one project.

Inspired by Databricks' **Predictive Query Execution (PQE)**, a
Databricks SQL Serverless feature that monitors query execution in
real time and replans mid-flight when it detects skew, spilling, or
inefficiency. PQE-lite deliberately does **not** try to replicate that
mid-flight replanning — because Postgres structurally can't do it (see
below). Instead, it does everything PQE does, but entirely *before*
the query runs, using an ML classifier trained on real query telemetry.

## Table of contents

- [Why this exists](#why-this-exists)
- [Why Postgres, not Spark](#why-postgres-not-spark)
- [Architecture](#architecture)
- [Key design decisions](#key-design-decisions-and-the-corrections-behind-them)
- [Repo structure](#repo-structure)
- [Setup](#setup)
- [Training the classifier](#training-the-classifier)
- [Known limitations, stated plainly](#known-limitations-stated-plainly)

---

## Why this exists

Most ML portfolio projects train on business/tabular data — churn,
fraud, sentiment. This one trains on **infrastructure telemetry**:
query execution plans and table statistics. That's a rarer and more
systems-flavored skill, and it's the project's primary differentiator.

The secondary differentiators, deliberately kept secondary rather than
the headline:
- **Systems/DB internals depth** — reading `EXPLAIN` plans, cost
  estimation, MVCC, autovacuum — is the credibility layer that makes
  the ML meaningful (the features mean something specific, not just
  "whatever a CSV happened to contain").
- **LLM reasoning beyond a chatbot wrapper** — the LLM here explains a
  decision that's already been made, grounded in retrieved reference
  notes. It's a *narrator*, not a decision-maker, and that boundary is
  deliberate (see below).

## Why Postgres, not Spark

Databricks needed to build PQE as a distinct feature specifically
because Spark's older **Adaptive Query Execution (AQE)** could only
replan *between* stages, not mid-stage — PQE closes that gap by
reacting to spill/skew signals in real time, mid-flight.

**Postgres can't do either of those things.** Once a query's plan is
chosen, Postgres runs it to completion — there is no hook for
mid-execution replanning at all. So every "adaptive" decision in this
project has to happen **before** the query runs, using only cheap,
pre-flight signals. That constraint shapes the entire architecture
below, and it's the reason this project is honestly framed as
*predictive plan-correction*, not *adaptive execution* — the two are
architecturally different problems, and conflating them would be the
easiest way to get caught out in an interview.

## Architecture

```
query submitted
  -> EXPLAIN (pre-flight, no execution)          [Postgres, cheap]
  -> stats & column-index coverage check          [Postgres, cheap]
  -> ML classifier: P(cost estimate is wrong)      [scikit-learn/XGBoost]
       -> apply_intervention   (misestimate likely: ANALYZE / flag missing index)
       -> warn_only            (cost is high but estimate looks correct --
                                 e.g. a genuine cross join with no filter)
       -> no_action            (low misestimate risk)
  -> execute query once (the ONLY real execution in the whole pipeline)
  -> LLM explains the decision, grounded in retrieved reference notes
  -> log outcome (written to MongoDB, shape varies by branch)
```

Orchestrated with **LangGraph**, specifically because there's a
genuine branch point with fan-in — three paths reconverge before
execution — which a linear LangChain chain doesn't model well.
**LangChain** handles the LLM call and prompt templating inside the
explanation node. Postgres access goes through **psycopg3** with a
connection pool (`psycopg_pool.ConnectionPool`).

Everything before "execute query once" is cheap and reversible —
`EXPLAIN`, stats lookups, a classifier prediction, maybe a lightweight
fix. The expensive thing only ever runs once, at the end, after all
the deciding is done.

## Key design decisions, and the corrections behind them

Each of these came from a real mistake caught and fixed during
development — kept here because the reasoning behind a decision is
worth more than the decision itself in an interview.

**Plain `EXPLAIN`, never `EXPLAIN ANALYZE`, in the online pipeline.**
`EXPLAIN ANALYZE` actually *executes* the query to get real timings —
running it "for a preview" before the real query would mean running
the expensive query twice. `db.explain_query()` (online) and
`db.explain_analyze_query()` (offline training only) are kept
deliberately separate so this distinction can't blur later.

**Concurrent `EXPLAIN` calls are safe.** Postgres uses MVCC: reads
don't block reads or writes. Only DDL (`CREATE INDEX` without
`CONCURRENTLY`, `ALTER TABLE`, `VACUUM FULL`) takes conflicting locks.

**Stats don't update live.** They refresh only on manual `ANALYZE` or
when autovacuum's threshold trips (~10% + 50 rows changed, by
default) — confirmed as a real, not just theoretical, issue: an
early real training batch had `seconds_since_last_analyze` computed
as ~1.78 billion (a raw Unix-epoch fallback) because the TPC-H tables
had *never* been analyzed even once. Running `ANALYZE` and
re-collecting turned that into real, small, meaningful values (59 to
~25,000 seconds) — see the log-transform section below for what that
scale jump then required.

**The classifier predicts "is the estimate wrong," not "is this query
slow."** A correctly-estimated, genuinely expensive cross join gets
`warn_only`, not `apply_intervention` — there's nothing to fix. The
label is a symmetric log-ratio between estimated and actual rows
(`compute_label` in `classifier.py`), not a raw difference, so being
off 10x in either direction is treated the same way.

**Column-level index coverage, not table-level.** The original design
checked "does this table have ANY index" — misleading, since a
primary-key-only index reports `True` even when the actual filtered
column has zero coverage. `column_analysis.py` replaced this with a
signal that matches the exact columns a query references in its
`WHERE`/`JOIN`/`GROUP BY`/`ORDER BY` against real indexed columns via
`information_schema`. A real end-to-end test (deliberately mixed
index coverage across a 3-table join) caught a genuine bug in the
first version of this check — it verified *any* overlap between
referenced and indexed columns instead of *full* coverage, so a
`GROUP BY (a, b)` with only `a` indexed incorrectly passed. Fixed to a
proper subset check. Kept alongside a second, complementary,
zero-extra-query signal (`filtered_seq_scan_count`/`index_scan_count`,
read straight from the plan's chosen scan node) — the two catch
different things: the plan-derived one reflects what the planner
actually did at execution time; the column-derived one is independent
of that choice and is the only one that catches `GROUP BY`/`ORDER BY`.

**The risk threshold is chosen empirically, never a hardcoded guess
and never a median.** A median describes typical traffic, not risk —
calling the median "high risk" would flag half of normal queries.
`scripts/train_classifier.py`'s `choose_threshold()` picks the lowest
threshold that still meets a minimum precision floor on held-out
predictions.

**Model selection uses cross-validation, not a fixed train/test/val
split, at this data size.** At N≈30, a 70/30 split evaluates on ~9
rows, and a three-way split leaves almost nothing per bucket.
`model_evaluation.py` uses stratified k-fold CV instead — every row
gets evaluated exactly once by a model that never saw it in that
fold's training. Three models (logistic regression, a calibrated
random forest, calibrated XGBoost) are compared the same way, and the
winner is picked by cross-validated ROC AUC — not assumed. Random
forest/XGBoost are wrapped in `CalibratedClassifierCV` specifically
because raw tree-ensemble `predict_proba()` is well known to be poorly
calibrated (it's "fraction of trees voting," not a real probability),
which matters directly here since the three-way branch depends on the
probability being meaningful, not just the predicted class.

**SMOTE, if used, is wired leakage-safely.** Applying SMOTE before
splitting can put a synthetic point and the real point it was
interpolated from into different CV folds, so a model gets evaluated
on a near-duplicate of something it already trained on — inflated
metrics for the wrong reason. `model_evaluation.py` wires SMOTE
through `imblearn`'s `Pipeline` so it only ever resamples inside each
fold's training side. Off by default (`PQE_USE_SMOTE=false`); even
done safely, it doesn't add real information, only interpolates
between existing minority-class examples — real data collection is
the actual fix for small N, not this.

**Log-transforming multi-order-of-magnitude features.**
`estimated_cost`, `estimated_rows`, and `seconds_since_last_analyze`
get `log1p()` applied before scaling (`classifier.py`'s
`_vectorize()`). Confirmed against real data: merging a
never-analyzed batch with a freshly-analyzed one made
`seconds_since_last_analyze` span 59 to 1.78 billion — raw,
`StandardScaler`'s mean/std get dominated by the extreme end, which
squashes the small-scale variation that actually matters down to near
nothing. After `log1p`, the same data spans a 5.2-fold range instead.
Confirmed effect: every model's cross-validated ROC AUC improved
substantially after combining more real data *and* applying this
transform (logistic 0.746 → 0.867, random_forest 0.761 → 0.890,
xgboost 0.716 → 0.938).

**The model gets evaluated before it gets saved, and the evaluation is
saved too.** `collect_training_data.py` runs `compare_models()`,
prints the comparison, picks a winner, refits it on the full dataset,
and writes a `*.metrics.json` file alongside the model with an
explicit `"quality"` field (`"provisional"` or `"unreliable"`) — so a
low-quality model, expected at small N, is saved and clearly labeled,
not silently trusted or silently blocked.

**The LLM explains, it doesn't decide.** By the time it runs, the
branch and any intervention have already happened, using the
classifier's output. It's grounded in a small, fixed, hand-written set
of Postgres reference notes (`rag.py`) instead of open web search —
reproducible and not dependent on what today's search results happen
to say, at the cost of not covering topics outside that small note set.

**MongoDB is used for exactly one thing, for a specific reason.** The
three branches produce genuinely different-shaped decision records —
`intervene` carries actions and staleness numbers, `warn_only` carries
cost/cross-join context, `no_action` carries almost nothing. That's a
real "the data shape justifies the database choice," not a résumé
keyword: every fixed-shape, structured signal (plans, stats, features)
stays in Postgres; only the variable-shape decision log goes to Mongo.

## Repo structure

```
pqe_lite/
  config.py           Every tunable value (PQE_* env vars), one place
  db.py                Postgres access: pooled psycopg3, plain EXPLAIN
                        (online) + EXPLAIN ANALYZE (offline training only)
  features.py          Plan -> QueryFeatures. Multi-table aggregation,
                        plan-derived scan-type signals
  column_analysis.py    Column-level index coverage (has_relevant_index)
  classifier.py         MisestimateClassifier: fit/predict/save/load,
                        label definition, log-transform, FEATURE_ORDER
  model_evaluation.py   Cross-validated comparison across logistic /
                        random_forest / xgboost, optional leakage-safe SMOTE
  intervention.py       The three-way branch + apply_intervention
  rag.py                Hand-written Postgres reference notes + retrieval
  llm_explain.py        LangChain call, grounded in rag.py, with retry
                        + fallback
  mongo_log.py          Variable-shape decision log
  query_loader.py       Parses "-- Qxx" labeled .sql files
  graph.py              LangGraph wiring, node names match the
                        architecture diagram
  logging_setup.py      Structured logging config
  main.py               Entry point

scripts/
  tpc-h-queries.sql             The 22 standard TPC-H queries
  extra_queries.sql             5 added misestimate-stress queries
  optional_expensive_queries.sql   NOT run by default -- a real,
                                    slow, unfiltered cross join
  collect_training_data.py     Runs every query via EXPLAIN ANALYZE,
                                builds features+labels, compares
                                models, saves the winner + metrics
  train_classifier.py          choose_threshold() + build_training_set()
                                for historical-run dicts from elsewhere
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a running Postgres instance, a running MongoDB instance, and
an `ANTHROPIC_API_KEY` environment variable for the explanation layer.
See `config.py` for the full list of `PQE_*` environment variables —
everything has a local-dev default, so it runs against `localhost` out
of the box if both databases are up.

## Training the classifier

```bash
python -m scripts.collect_training_data
```

This runs the 22 standard TPC-H queries plus 5 supplementary
misestimate-stress queries (`extra_queries.sql` — a function-wrapped
predicate, correlated columns, a cheap cross join to test routing, a
cross-table `OR`, and a 4-way join), each once via `EXPLAIN ANALYZE`,
builds labeled features, cross-validates all three models, and saves
the winner plus a `*.metrics.json` file. `scripts/optional_expensive_queries.sql`
is excluded from this by default — it contains a real, unfiltered
`lineitem, nation` cross join (~1.5 billion row combinations at SF10);
run it manually only if you're prepared to wait.

**To grow the dataset meaningfully rather than just re-running the
same state:** re-run the collector after genuinely changing data
state — before/after a manual `ANALYZE`, before/after dropping or
adding an index, after some real writes accumulate. Each of those
gives the classifier a real contrast to learn from; running the exact
same script against the exact same unchanged database again does not.

## Known limitations, stated plainly

- **Training data is still small.** Even combined across two
  collection runs, N is in the tens, not hundreds. Cross-validated
  metrics are legitimate at this size, but treat any single run's
  numbers as provisional until the dataset grows further — check each
  model's `"quality"` field in its metrics file.
- **`n_mod_since_analyze` has no variance yet.** Both real collection
  runs so far happened immediately after an `ANALYZE`, before any
  writes could accumulate. A third collection pass, done after some
  real inserts/updates, is needed to give this feature real signal.
- **Mongo decision logs can't be used to retrain yet.**
  `train_classifier.py`'s `load_historical_runs_from_mongo()` is not
  wired up to work — the Mongo log stores decision summaries, not the
  raw plan/stats `build_training_set()` needs. Flagged in that
  function's docstring rather than silently broken.
- **`filter_columns_indexed`-style coverage doesn't extend to
  expression indexes or partial indexes.** `column_analysis.py`
  matches simple column names; a query using a function-wrapped
  predicate that's covered by an expression index would currently be
  under-credited as "not indexed."
