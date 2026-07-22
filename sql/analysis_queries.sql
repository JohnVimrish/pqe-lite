


SELECT pid, usename, datname, state, query, query_start
from pg_stat_activity
WHERE state = 'active';


SELECT c_nationkey, count(*)
from --ai_ml_experiment.lineitem
 ai_ml_experiment.customer
group by c_nationkey;

Select * from ai_ml_experiment.nation


 SELECT
        relname,
        n_live_tup,
        n_mod_since_analyze,
        last_analyze,
        last_autoanalyze,
        EXTRACT(EPOCH from (now() - GREATEST(
            COALESCE(last_analyze, 'epoch'::timestamp),
            COALESCE(last_autoanalyze, 'epoch'::timestamp)
        ))) AS seconds_since_last_analyze
    from pg_stat_user_tables
    WHERE relname


 Select * from pg_stat_user_tables;






EXPLAIN (ANALYZE, FORMAT JSON) SELECT *
from (select * from ai_ml_experiment.lineitem l where
    (extract( year from l.l_commitdate )  >= 1996 and extract( year from l.l_commitdate )  <= 1997))
   , (Select * from ai_ml_experiment.customer c where c_nationkey  >=20 Limit 10000)
where
1=1
         ai_ml_experiment.partsupp;


EXPLAIN (ANALYZE, FORMAT JSON) Select * from
(Select * from ai_ml_experiment.orders Limit 10000) a,
(SELECT * from ai_ml_experiment.partsupp LIMIT 3337709) b
where 1=1