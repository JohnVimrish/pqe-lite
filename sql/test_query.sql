SELECT pid, state, query, age(clock_timestamp(), query_start)
from pg_stat_activity
WHERE state != 'idle';

Select *
from information_schema.tabl
where table_type = 'BASE TABLE'
and not (table_schema like  'pg_%' or table_schema ='information_schema');






;
select * from ai_ml_experiment.lineitem l where
    (extract( year from l.l_commitdate )  >= 1996 and extract( year from l.l_commitdate )  <= 1997);


with mn as
(Select min(acctbal)                                            min_acctbal,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY acctbal) AS median_value,
       max(acctbal)                                            max_acctbal
FROM (Select *, sum(c_acctbal) over (partition by c_custkey,c_name order by c_custkey asc ) acctbal
      from ai_ml_experiment.customer
      where c_mktsegment = 'HOUSEHOLD') ls) ,
hm as  (
Select  c.*  from
   ai_ml_experiment.customer c, mn as mn
where 1=1 and
      c.c_acctbal < mn.median_value
limit 5000)
SELECT mn.*
from (select * from ai_ml_experiment.lineitem l where
    (extract( year from l.l_commitdate )  >= 1996 and extract( year from l.l_commitdate )  <= 1997)) mn
   , hm
where
1=1
