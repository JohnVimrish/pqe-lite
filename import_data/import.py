import os
import re
import psycopg
from psycopg import sql

# Database connection string
DB_CONN_STRING = "postgresql://user_one:abc$12345@localhost:5432/ai_experiment"

# Directory where your TPC-H SF10 CSV files are stored
CSV_DIRECTORY = "F:/snowflake_sample_schema/"
SCHEMA_NAME = "ai_ml_experiment" 

# Cleaned Schema Map: Removed hardcoded schema prefixes to rely on session search_path
TPCH_SCHEMA = {
    "part": {
        "columns": "p_partkey INT, p_name VARCHAR(55), p_mfgr CHAR(25), p_brand CHAR(10), p_type VARCHAR(25), p_size INT, p_container CHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23)",
        "indexes": ["CREATE UNIQUE INDEX idx_part_pkey ON part(p_partkey);"]
    },
    "supplier": {
        "columns": "s_suppkey INT, s_name CHAR(25), s_address VARCHAR(40), s_nationkey INT, s_phone CHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101)",
        "indexes": [
            "CREATE UNIQUE INDEX idx_supplier_pkey ON supplier(s_suppkey);",
            "CREATE INDEX idx_supplier_nationkey ON supplier(s_nationkey);"
        ]
    },
    "partsupp": {
        "columns": "ps_partkey INT, ps_suppkey INT, ps_availqty INT, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199)",
        "indexes": [
            "CREATE UNIQUE INDEX idx_partsupp_pkey ON partsupp(ps_partkey, ps_suppkey);",
            "CREATE INDEX idx_partsupp_suppkey ON partsupp(ps_suppkey);"
        ]
    },
    "customer": {
        "columns": "c_custkey INT, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INT, c_phone CHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment CHAR(10), c_comment VARCHAR(117)",
        "indexes": [
            "CREATE UNIQUE INDEX idx_customer_pkey ON customer(c_custkey);",
            "CREATE INDEX idx_customer_nationkey ON customer(c_nationkey);"
        ]
    },
    "orders": {
        "columns": "o_orderkey INT, o_custkey INT, o_orderstatus CHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority CHAR(15), o_clerk CHAR(15), o_shippriority INT, o_comment VARCHAR(79)",
        "indexes": [
            "CREATE UNIQUE INDEX idx_orders_pkey ON orders(o_orderkey);",
            "CREATE INDEX idx_orders_custkey ON orders(o_custkey);",
            "CREATE INDEX idx_orders_orderdate ON orders(o_orderdate);"
        ]
    },
    "lineitem": {
        "columns": "l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag CHAR(1), l_linestatus CHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstructions CHAR(25), l_shipmode CHAR(10), l_comment VARCHAR(44)",
        "indexes": [
            "CREATE UNIQUE INDEX idx_lineitem_pkey ON lineitem(l_orderkey, l_linenumber);",
            "CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);",
            "CREATE INDEX idx_lineitem_partsupp ON lineitem(l_partkey, l_suppkey);",
            "CREATE INDEX idx_lineitem_shipdate ON lineitem(l_shipdate);"
        ]
    },
    "nation": {
        "columns": "n_nationkey INT, n_name CHAR(25), n_regionkey INT, n_comment VARCHAR(152)",
        "indexes": ["CREATE UNIQUE INDEX idx_nation_pkey ON nation(n_nationkey);"]
    },
    "region": {
        "columns": "r_regionkey INT, r_name CHAR(25), r_comment VARCHAR(152)",
        "indexes": ["CREATE UNIQUE INDEX idx_region_pkey ON region(r_regionkey);"]
    }
}

def migrate_data():
    if not os.path.exists(CSV_DIRECTORY):
        print(f"Error: Directory '{CSV_DIRECTORY}' does not exist.")
        return

    # Establish connection to PostgreSQL
    with psycopg.connect(DB_CONN_STRING, autocommit=False) as conn:
        with conn.cursor() as cur:
            
            # 1. Safely handle Schema initialization using psycopg.sql utilities
            print(f"Re-creating schema: {SCHEMA_NAME}")
            cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE;").format(sql.Identifier(SCHEMA_NAME)))
            cur.execute(sql.SQL("CREATE SCHEMA {};").format(sql.Identifier(SCHEMA_NAME)))
            
            # 2. Set the session's search path to our schema. 
            # This completely removes the need to append '{SCHEMA_NAME}.' to tables or indexes!
            cur.execute(sql.SQL("SET search_path TO {};").format(sql.Identifier(SCHEMA_NAME)))
            
            # Walk through CSV folder and match files to schemas
            for file_name in os.listdir(CSV_DIRECTORY):
                if not file_name.endswith('.csv'):
                    continue
                
                # Extract table name from filename (e.g., "lineitem.csv" -> "lineitem")
                table_name = re.sub(r'\.csv$', '', file_name).lower()
                
                if table_name not in TPCH_SCHEMA:
                    print(f"Skipping file {file_name}: No matching TPC-H schema definition found.")
                    continue
                
                file_path = os.path.join(CSV_DIRECTORY, file_name)
                schema_info = TPCH_SCHEMA[table_name]
                
                print(f"\n--- Processing Table: {table_name} ---")
                
                # 3. Create Table inside the active schema context
                print(f"Creating table: {table_name}")
                cur.execute(f"CREATE TABLE {table_name} ({schema_info['columns']});")
                
                # 4. Fast Copy Data Bulk Upload via STDIN stream
                print(f"Streaming data from {file_name} via COPY...")
                copy_query = f"""
                    COPY {table_name} 
                    from STDIN 
                    WITH (FORMAT CSV, HEADER true, QUOTE '"', DELIMITER ',');
                """
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    with cur.copy(copy_query) as copy:
                        while data := f.read(65536):  # 64KB block iterations
                            copy.write(data)
                            
                print(f"Data ingestion completed for {table_name}.")
                
                # 5. Generate Indexes (Post-Copy Execution)
                print(f"Generating indexes for {table_name}...")
                for index_query in schema_info["indexes"]:
                    cur.execute(index_query)
                    
            # Commit the entire batch sequence safely
            conn.commit()
            print("\nMigration Completed Successfully!")

if __name__ == "__main__":
    migrate_data()