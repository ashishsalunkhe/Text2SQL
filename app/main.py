# Directory Structure:
# mimic_text_to_sql/
# ├── data/
# │   ├── csv/               ← All CSVs here
# │   ├── mimic_iii.db       ← DB will be created here
# │   └── schema_map.json    ← Generated schema info
# └── app.py                ← This script

import os
import sqlite3
import json
import pandas as pd
import openai
import faiss
import numpy as np
from tqdm import tqdm
import argparse

# === STEP 1: Path Setup ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "mimic_iii.db")
SCHEMA_JSON_PATH = os.path.join(DATA_DIR, "schema_map.json")

# ✅ Create DB directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# === STEP 2: Load CSVs into SQLite if DB doesn't exist ===
if not os.path.exists(DB_PATH):
    print("📦 Creating mimic_iii.db from CSVs...")
    conn = sqlite3.connect(DB_PATH)
    for file in tqdm(os.listdir(CSV_DIR)):
        if file.endswith(".csv"):
            table = file.replace(".csv", "").lower()
            df = pd.read_csv(os.path.join(CSV_DIR, file), low_memory=False)
            df.to_sql(table, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("✅ Database created at", DB_PATH)
else:
    print("✅ Using existing SQLite DB at", DB_PATH)

# === STEP 3: Extract Schema and Save JSON ===
if not os.path.exists(SCHEMA_JSON_PATH):
    print("🔍 Extracting schema to JSON map...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        schema_info[table] = {
            "columns": [col[1] for col in columns],
            "types": {col[1]: col[2] for col in columns}
        }

    # Annotate known joins
    known_joins = {
        "prescriptions": ["subject_id", "hadm_id"],
        "diagnoses_icd": ["subject_id", "hadm_id", "icd9_code"],
        "procedures_icd": ["subject_id", "hadm_id"],
        "microbiologyevents": ["subject_id"],
        "icustays": ["subject_id", "hadm_id", "icustay_id"],
        "patients": ["subject_id", "hospital_expire_flag"],
        "labevents": ["subject_id", "itemid"],
        "d_labitems": ["itemid", "label"],
        "d_icd_diagnoses": ["icd9_code", "long_title"]
    }

    for table, keys in known_joins.items():
        if table in schema_info:
            schema_info[table]["join_keys"] = keys

    with open(SCHEMA_JSON_PATH, "w") as f:
        json.dump(schema_info, f, indent=2)
    print("✅ Schema map saved to", SCHEMA_JSON_PATH)
else:
    print("✅ Using existing schema map at", SCHEMA_JSON_PATH)

# === STEP 4: Prompt & Query ===
def format_schema(schema_json):
    return "\n\n".join(
        [f"Table: {tbl}\nColumns: {', '.join(meta['columns'])}" for tbl, meta in schema_json.items()]
    )

def build_prompt(user_question, schema_json):
    context = """
Helpful Notes:
- Use subject_id or hadm_id to link patient-level tables.
- diagnoses_icd.icd9_code LIKE '250%' means diabetes.
- patients.hospital_expire_flag = 1 means patient died.
- microbiologyevents.org_name is for organisms.
- d_labitems.label gives lab test names via labevents.itemid = d_labitems.itemid
- labevents + icustays join via subject_id or hadm_id
"""
    return f"""
You are a medical SQL assistant.

{context}

Schema:
{format_schema(schema_json)}

User question:
"{user_question}"

Return only the SQLite SQL query.
"""

# === STEP 5: Run GPT + SQL ===
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_sql_from_gpt(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res["choices"][0]["message"]["content"].strip().replace("```sql", "").replace("```", "").strip()

def run_sql(query):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    conn.close()
    return pd.DataFrame(rows, columns=cols)

# === STEP 6: Main CLI Logic ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-powered SQL over MIMIC-III data.")
    parser.add_argument("--question", type=str, required=True, help="Natural language question to convert into SQL")
    parser.add_argument("--print_only", action="store_true", help="Only print the SQL without executing it")
    args = parser.parse_args()

    with open(SCHEMA_JSON_PATH) as f:
        schema_data = json.load(f)

    prompt = build_prompt(args.question, schema_data)
    print("\n📤 Prompt Sent to GPT:\n", prompt)

    sql_query = get_sql_from_gpt(prompt)
    print("\n🧠 Generated SQL:\n", sql_query)

    if not args.print_only:
        try:
            result_df = run_sql(sql_query)
            print("📊 Query Result Preview:")
            print(result_df.head())

            # ✅ Logging
            log_path = os.path.join(DATA_DIR, "query_log.csv")
            log_entry = {
                "question": args.question,
                "sql_query": sql_query,
                "result_preview": result_df.head().to_json(orient="records")
            }
            if os.path.exists(log_path):
                pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=False, index=False)
            else:
                pd.DataFrame([log_entry]).to_csv(log_path, mode='w', header=True, index=False)

        except Exception as e:
            print("❌ SQL execution failed:", e)