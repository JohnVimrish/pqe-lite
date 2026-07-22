import os
import subprocess
import sys
import psycopg
from psycopg import sql # Based on your table metadata, assuming PostgreSQL

# --- CONFIGURATION ---
DB_CONN_STRING = ""

# Path to the next script you want to trigger
NEXT_SCRIPT_PATH = "f:/code_experiment/src/collect_training_data.py"
TARGET_SCHEMA = "ai_ml_experiment"  # Change this to your schema name


def run_database_analyze():
    print("🔄 Connecting to the database...")
    try:
        # Establish connection
        conn =  psycopg.connect(DB_CONN_STRING)
        # Postgres requires autocommit=True to run database-wide operations like ANALYZE
        conn.autocommit = True
        cursor = conn.cursor()

        print(f"⚡ Fetching all tables under schema '{TARGET_SCHEMA}'...")
        cursor.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_type = 'BASE TABLE';
        """,
            (TARGET_SCHEMA,),
        )

        tables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Found {len(tables)} tables in schema '{TARGET_SCHEMA}'.")

        print(f"🔄 Analyzing {len(tables)} tables in '{TARGET_SCHEMA}'...")
        for table in tables:
            # Use double quotes around schema and table names to handle odd characters safely
            cursor.execute(f'ANALYZE "{TARGET_SCHEMA}"."{table}";')
            print(f"  ✓ Analyzed {table}")

        print("✅ Schema statistics successfully updated!")


        print("✅ Database statistics successfully updated!")
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Database error: {e}")
        print("Stopping pipeline. Will not run the next script.")
        sys.exit(1)


def trigger_next_script():
    print(f"\n🚀 Triggering next script: {os.path.basename(NEXT_SCRIPT_PATH)}")

    if not os.path.exists(NEXT_SCRIPT_PATH):
        print(f"❌ Error: Could not find script at {NEXT_SCRIPT_PATH}")
        sys.exit(1)

    try:
        # sys.executable keeps it safely inside your active virtual environment
        result = subprocess.run(
            [sys.executable, NEXT_SCRIPT_PATH],
            check=True,  # Crashes this script if the called script fails
            text=True,  # Ensures terminal output prints properly
        )
        print("🎉 Entire pipeline completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ The script {os.path.basename(NEXT_SCRIPT_PATH)} failed.")
        sys.exit(e.returncode)


if __name__ == "__main__":
    # 1. Run the database optimization first
    run_database_analyze()

    # 2. Immediately kick off your training data collection script right after
    trigger_next_script()