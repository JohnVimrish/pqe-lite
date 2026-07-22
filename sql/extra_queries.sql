-- Supplementary queries, not part of the standard TPC-H 22.
-- Each one is chosen to exercise a specific misestimate mechanism
-- discussed in the design docs, and deliberately scoped (date ranges,
-- small tables) to stay cheap to run at SF10 -- none of these should
-- take more than a few seconds.

-- QX01: function on a column defeats normal index usage, a common
-- real-world cause of planner misestimates (see rag.py's
-- cost_estimate note).
SELECT o_orderkey, o_orderdate
from ai_ml_experiment.orders
WHERE EXTRACT(YEAR from o_orderdate) = 1996;

-- QX02: correlated columns (l_shipdate = l_commitdate is rare but not
-- independent the way the planner assumes -- see rag.py's
-- correlated_columns note). Restricted to one month to stay cheap.
SELECT count(*)
from ai_ml_experiment.lineitem
WHERE l_shipdate = l_commitdate
  AND l_shipdate >= DATE '1996-01-01'
  AND l_shipdate < DATE '1996-02-01';

-- QX03: a genuine, cheap CROSS JOIN between two small dimension
-- tables. Used to verify the is_cross_join routing works correctly
-- even when the cost happens to be low -- see intervention.py's
-- decide_branch, which routes ANY detected cross join to warn_only
-- regardless of probability.
SELECT n.n_name, r.r_name
from ai_ml_experiment.nation n
CROSS JOIN ai_ml_experiment.region r;

-- QX04: OR across columns from two different tables in a join --
-- stresses the planner's column-independence assumption across
-- tables, not just within one.
SELECT count(*)
from ai_ml_experiment.customer c
JOIN ai_ml_experiment.orders o ON c.c_custkey = o.o_custkey
WHERE c.c_mktsegment = 'AUTOMOBILE'
   OR o.o_orderpriority = '1-URGENT';

-- QX05: a 4-way join with a narrow date filter -- bumps table_count
-- while staying cheap, to give the classifier examples of
-- misestimate risk that scales with join width (see features.py's
-- table_count feature).
SELECT count(*)
from ai_ml_experiment.customer c
JOIN ai_ml_experiment.orders o ON c.c_custkey = o.o_custkey
JOIN ai_ml_experiment.lineitem l ON o.o_orderkey = l.l_orderkey
JOIN ai_ml_experiment.nation n ON c.c_nationkey = n.n_nationkey
WHERE o.o_orderdate >= DATE '1996-01-01'
  AND o.o_orderdate < DATE '1996-02-01';

-- QX06: a leading-wildcard LIKE on a free-text column. Postgres's
-- planner is notoriously bad at estimating selectivity for '%...%'
-- patterns without extended statistics -- a good, realistic example
-- of the "estimate is wrong" case rather than the "genuinely
-- expensive" case. This does a full scan of lineitem (~60M rows at
-- SF10), so expect it to take longer than the others here -- still
-- bounded, just not instant.
SELECT count(*)
from ai_ml_experiment.lineitem
WHERE l_comment LIKE '%express%deliver%';

-- QX07: two predicates on columns within the same table that are not
-- actually independent in practice (part size and container size tend
-- to move together), which the planner's default statistics assume
-- they are. Cheap to run (part is small), and a second, distinct
-- example of the correlated-columns misestimate mechanism alongside
-- QX02.
SELECT count(*)
from ai_ml_experiment.part
WHERE p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND p_size < 10;


