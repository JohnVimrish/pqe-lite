-- NOT run by collect_training_data.py by default -- run these
-- manually, one at a time, if you specifically want a real example of
-- "genuinely expensive AND correctly estimated" for the classifier.
--
-- WARNING: QX_EXPENSIVE_01 is a real, unfiltered cross join between
-- lineitem (~60M rows at SF10) and nation (25 rows) -- roughly 1.5
-- billion row combinations. Postgres still has to do that join work
-- even for a bare count(*). This can run for a long time depending on
-- your hardware; there is no LIMIT that helps, since a cross join has
-- no condition to short-circuit on. Only run this if you're prepared
-- to wait, or want to demonstrate the "cost estimate is correct, this
-- is just expensive" case concretely rather than with the small,
-- cheap nation/region cross join in extra_queries.sql (QX03).

-- QX_EXPENSIVE_01
SELECT *
from (select * from ai_ml_experiment.lineitem l where
    (extract( year from l.l_commitdate )  >= 1996 and extract( year from l.l_commitdate )  <= 1997))
   , (Select * from ai_ml_experiment.customer c where c_nationkey  >=20 Limit 10000)
where
1=1




-- QX_EXPENSIVE_02
Select * from
(Select * from ai_ml_experiment.orders Limit 10000) a,
(SELECT * from ai_ml_experiment.partsupp LIMIT 3337709) b
where 1=1