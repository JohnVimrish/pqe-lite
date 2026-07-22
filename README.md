# PQE-lite

A predictive plan-correction layer for Postgres — an ML classifier
trained on real query telemetry, orchestrated with LangGraph, that
decides *before* a query runs whether Postgres's cost estimate is
likely wrong, and either fixes it, warns about it, or leaves it alone.
An LLM narrates each decision in plain language, grounded in retrieved
reference notes rather than free-form guessing.

Inspired by Databricks' **Predictive Query Execution (PQE)** — a
feature that monitors query execution in real time and replans
mid-flight when it detects skew or spilling. PQE-lite doesn't try to
replicate that mid-flight replanning, because Postgres structurally
can't do it (more on that below). Instead, it does everything PQE
does, but entirely *before* the query runs.

## Table of contents

- [Why this exists](#why-this-exists)
- [Why Postgres can't replan mid-flight](#why-postgres-cant-replan-mid-flight)
- [Workflow](#workflow)
- [Real output](#real-output)
- [Key design decisions](#key-design-decisions)
- [Repo structure](#repo-structure)
- [Setup](#setup)
- [Security note](#security-note)
- [Known limitations](#known-limitations-stated-plainly)

---

## Why this exists

Most ML portfolio projects train on business/tabular data — churn,
fraud, sentiment. This one trains on **infrastructure telemetry**:
query execution plans and table statistics. That's a rarer skill
combination, and it's the project's primary differentiator — it
bridges three things that rarely show up together in one project:

- **ML on operational data** — a classifier trained on `EXPLAIN` plan
  shape and table staleness, not a CSV of business records.
- **Database internals depth** — the ML only means something because
  the features underneath it (cost estimates, MVCC, autovacuum,
  column-level index coverage) are grounded in how Postgres actually
  works, not treated as an opaque black box.
- **LLM reasoning beyond a chatbot wrapper** — the LLM here explains a
  decision that's already been made deterministically, grounded in a
  small fixed set of reference notes. It's a narrator, not a
  decision-maker — see [Key design decisions](#key-design-decisions).

## Why Postgres can't replan mid-flight

Spark's Adaptive Query Execution could only replan *between* stages;
Databricks built PQE specifically to react to skew/spill signals
*mid-stage*, in real time. Postgres has no equivalent hook at all —
once a plan is chosen, it runs to completion. So every "adaptive"
decision here has to happen *before* the query runs, using only cheap
pre-flight signals. That constraint shapes the whole architecture, and
it's why this is framed as *predictive plan-correction*, not *adaptive
execution* — the two are architecturally different problems.

## Workflow

```mermaid
flowchart TD
    A[Query submitted] --> B["EXPLAIN (pre-flight, no execution)"]
    B --> C[Check table stats + column-level index coverage]
    C --> D{ML classifier:<br/>P cost estimate is wrong}
    D -->|misestimate likely| E[Apply intervention<br/>ANALYZE / flag missing index]
    D -->|cost is high, estimate looks correct| F[Warn only<br/>e.g. a genuine cross join]
    D -->|low risk| G[No action]
    E --> H[Execute query once]
    F --> H
    G --> H
    H --> I[LLM explains the decision<br/>grounded in reference notes]
    I --> J[Log outcome to MongoDB]
```

Everything before "Execute query once" is cheap and reversible —
`EXPLAIN`, stats lookups, a classifier prediction, maybe a lightweight
fix. The expensive thing only ever runs once, after all the deciding
is done. Orchestrated with **LangGraph** specifically because of that
branch-then-fan-in shape — three paths reconverge before execution,
which a linear LangChain chain doesn't model well.

## Real output

These are unedited explanations the pipeline produced against a live
TPC-H dataset — included here as evidence the reasoning is grounded in
real plan data, not templated filler.

**Catching a likely bug, not just a missing index** — this is the
result worth leading with:

> *"The plan reveals an unconditioned cross join between the `mn` and
> `hm` subqueries. Per the relevant reference note, you should verify
> if an intended join key exists between these sides, as this often
> indicates a missing join predicate rather than an intended full
> product."*

**Intervening, with the actual numbers behind the decision:**

> *"I have decided to intervene because the query optimizer shows a
> high misestimate probability of 0.68 for the `lineitem` table.
> Despite having an index and recent statistics, the planner estimated
> 833,173 rows with a cost of 358,072.69 ... the estimation error
> suggests the plan may be suboptimal."*

**Correctly declining to recommend an index that wouldn't help:**

> *"The query performs a full aggregation on `lineitem` without any
> filters ... Because there are no filter conditions, the recommended
> composite index on `(l_linestatus, l_returnflag)` would not provide
> a performance gain ... a composite index only helps if the query's
> filter columns match a left-to-right prefix of the index."*

That last one matters more than it looks — it shows the system
reasoning about when a fix *doesn't* apply, not just pattern-matching
"missing index → recommend index."

## Key design decisions

The full reasoning behind each of these — including the real mistakes
that led to them — is worth reading if you're evaluating this project
technically, not just skimming it:

- **Plain `EXPLAIN`, never `EXPLAIN ANALYZE`, in the online pipeline**
  — `EXPLAIN ANALYZE` executes the query; running it as a "preview"
  would mean running the expensive query twice.
- **Column-level index coverage, not table-level** — checking "does
  this table have an index" is misleading; a primary-key-only index
  reports `True` even when the actually-filtered column has zero
  coverage. `column_analysis.py` matches the exact columns referenced
  in filters/joins/`GROUP BY`/`ORDER BY` against real indexed columns.
- **The classifier predicts "is the estimate wrong," not "is this
  query slow"** — a correctly-estimated, genuinely expensive cross
  join gets `warn_only`, not a fix, because there's nothing to fix.
- **Cross-validation, not a fixed train/test/val split**, at small N
  — a fixed split evaluates on a handful of rows; k-fold CV evaluates
  every row exactly once, using far more of a small dataset.
- **Model choice (logistic / random forest / XGBoost) is decided by
  cross-validated ROC AUC**, not preference — and tree-ensemble
  probabilities are calibrated (`CalibratedClassifierCV`) since raw
  `predict_proba()` from a forest is "fraction of trees voting," not a
  real probability — which matters here because the three-way branch
  depends on the probability meaning something.
- **The LLM explains, it doesn't decide** — grounded in a small,
  hand-written set of Postgres reference notes instead of open web
  search, so explanations stay reproducible.

## Repo structure

```
src/
  config.py, db.py, logging_setup.py     Infra: config, pooled psycopg3, logging
  features.py, column_analysis.py         Plan -> features, column-level index coverage
  classifier.py, model_evaluation.py      Model, cross-validated comparison
  intervention.py                          The three-way branch
  rag.py, llm_explain.py                  Grounded LLM explanation
  mongo_log.py                            Variable-shape decision log
  query_loader.py, collect_training_data.py, train_classifier.py
                                            Training data collection + selection
  graph.py, main.py                       LangGraph wiring + entry point

sql/            TPC-H queries + supplementary stress queries
schedule/       Scheduled ANALYZE + retraining trigger
import_data/    Dataset import utilities
test/           Test scaffolding
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (create one if it doesn't exist yet) and
set, at minimum:

```bash
NVIDIA_NIM_API_KEY=your-key-here      # rotate any key you've ever pasted elsewhere
PQE_SCHEDULE_DB_URL=postgresql://user:pass@localhost:5432/yourdb
```

Requires a running Postgres instance and a running MongoDB instance.
See `config.py` for the full list of tunable environment variables.

```bash
python src/collect_training_data.py   # train
python src/main.py                    # run the pipeline against a query
```

## Security note

This repo previously had a live API key and a database password
hardcoded in source (`config.py`, `schedule.py`) — both now read from
environment variables instead, and `.gitignore` excludes `.env` and
`models/*.joblib` metrics so this can't silently happen again. If
you're publishing this repo: **rotate any credential that was ever
hardcoded**, even if you don't think it made it into a commit — a key
pasted anywhere outside your own machine should be treated as
compromised.

## Known limitations, stated plainly

- Training data is still small (tens, not hundreds, of examples) —
  cross-validated metrics are legitimate at this size, but treat any
  single run's numbers as provisional; check each model's `"quality"`
  field in its saved metrics file.
- `n_mod_since_analyze` has had no variance across collection runs so
  far, since collection has always happened right after an `ANALYZE`.
- Mongo decision logs aren't wired up to retrain from yet — they store
  decision summaries, not the raw plan/stats needed to rebuild
  features.
- Column-level index matching doesn't yet extend to expression or
  partial indexes.
