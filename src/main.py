"""
Entry point -- Phase 2. Wires config, the Postgres pool, the Mongo
collection, and a trained (or stub) classifier into the graph.

This still won't produce meaningful decisions until the classifier is
trained on real historical data -- see classifier.py and
scripts/train_classifier.py.
"""

from __future__ import annotations

import db, mongo_log
from classifier import MisestimateClassifier, ClassifierNotTrainedError
from config import load_config
from graph import build_graph
from logging_setup import configure_logging, get_logger

logger = get_logger(__name__)



def run(sql: str, pg_pool: db.Pool) -> dict[str, any]:
    cfg = load_config()
    configure_logging(cfg.log_level)


    mongo_collection = mongo_log.get_collection(cfg.mongo)

    try:
        classifier = MisestimateClassifier.load(cfg.classifier.model_path)
    except FileNotFoundError:
        logger.warning(
            "no trained model found at %s -- using an untrained classifier "
            "(predict_proba will raise until you train one, see "
            "scripts/train_classifier.py)",
            cfg.classifier.model_path,
        )
        classifier = MisestimateClassifier()

    graph = build_graph()
    initial_state = {
        "sql": sql,
        "pg_pool": pg_pool,
        "mongo_collection": mongo_collection,
        "classifier": classifier,
        "config": cfg,
    }

    try:
        return graph.invoke(initial_state)
    except ClassifierNotTrainedError:
        logger.error(
            "cannot run the pipeline: classifier has not been trained yet. "
            "Run scripts/train_classifier.py against historical query runs first."
        )
        raise


def _get_databse_connection () :
    cfg = load_config()
    configure_logging(cfg.log_level)
    pg_pool = db.get_pool(cfg.postgres)
    return pg_pool

def _close_database_connection(pg_pool) :
    pg_pool.close()
    logger.info("Postgres connection pool closed.")



if __name__ == "__main__":
#     example_sql =  f'''with mn as
#                         (Select min(acctbal)                                            min_acctbal,
#                             PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY acctbal) AS median_value,
#                             max(acctbal)                                            max_acctbal
#                         FROM (Select *, sum(c_acctbal) over (partition by c_custkey,c_name order by c_custkey asc ) acctbal
#                             from ai_ml_experiment.customer
#                             where c_mktsegment = 'HOUSEHOLD') ls) ,
#                         hm as  (
#                         Select  c.*  from
#                         ai_ml_experiment.customer c, mn as mn
#                         where 1=1 and
#                             c.c_acctbal < mn.median_value
#                         limit 50000)
#                         SELECT mn.*
#                         from (select * from ai_ml_experiment.lineitem l where
#                             (extract( year from l.l_commitdate )  >= 1996 and extract( year from l.l_commitdate )  <= 1997)) mn
#                         , hm
#                         where
#                         1=1
# '''

    db_connection  = _get_databse_connection()
    queries  = [
            f'''
        DELETE FROM ai_ml_experiment.lineitem
        WHERE l_shipdate BETWEEN '1996-01-01' AND '1996-06-30';
        ''',
        f'''
       SELECT *
        FROM ai_ml_experiment.lineitem
        WHERE l_shipdate BETWEEN '1996-01-01' AND '1996-06-30';
        ''',
                                        f'''
        SELECT c_name, c_acctbal
        FROM ai_ml_experiment.customer
        WHERE c_custkey = 12345;
        ''',
                f'''
        SELECT l_returnflag, l_linestatus, SUM(l_quantity), AVG(l_extendedprice)
        FROM ai_ml_experiment.lineitem
        GROUP BY l_returnflag, l_linestatus;
        '''
        ]
    for values in queries  :
        run(values, db_connection)
    _close_database_connection(db_connection)
