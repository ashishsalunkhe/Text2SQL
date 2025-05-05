import streamlit as st
import openai
import sqlite3
import pandas as pd
import json
import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "mimic_iii.db")
SCHEMA_JSON_PATH = os.path.join(DATA_DIR, "schema_map.json")
LOG_PATH = os.path.join(DATA_DIR, "query_log.csv")

# === Load schema ===
@st.cache_data
def load_schema():
    with open(SCHEMA_JSON_PATH) as f:
        return json.load(f)

# === Format prompt ===
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

# === Run GPT ===
def get_sql_from_gpt(prompt):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res["choices"][0]["message"]["content"].strip().replace("```sql", "").replace("```", "").strip()

# === Run SQL ===
def run_sql(query):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    conn.close()
    return pd.DataFrame(rows, columns=cols)

# === Log to CSV ===
def log_interaction(question, sql_query, result_df):
    log_entry = {
        "question": question,
        "sql_query": sql_query,
        "result_preview": result_df.head().to_json(orient="records")
    }
    if os.path.exists(LOG_PATH):
        pd.DataFrame([log_entry]).to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        pd.DataFrame([log_entry]).to_csv(LOG_PATH, mode='w', header=True, index=False)

# === Query History ===
def load_query_history():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["question", "sql_query", "result_preview"])

# === Streamlit UI ===
st.set_page_config(page_title="MIMIC-III SQL Explorer", layout="wide")
st.title("🩺 MIMIC-III Natural Language SQL Explorer")

# --- Query History Viewer ---
with st.expander("🕘 Query History", expanded=False):
    history_df = load_query_history()
    if not history_df.empty:
        history_df['result_preview'] = history_df['result_preview'].astype(str)
        display_df = history_df[["question", "sql_query", "result_preview"]].rename(columns={
            "question": "Question",
            "sql_query": "SQL Query",
            "result_preview": "Result"
        }).sort_index(ascending=False)
        st.dataframe(display_df)

        # Download button
        csv_download = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Query History as CSV", data=csv_download, file_name="query_history.csv", mime="text/csv")

        # Re-run query
        selected = st.selectbox("🔁 Re-run a previous query", display_df["Question"])
        rerun_row = display_df[display_df["Question"] == selected].iloc[0]
        st.code(rerun_row["SQL Query"], language="sql")
        try:
            df = run_sql(rerun_row["SQL Query"])
            st.dataframe(df)
        except Exception as e:
            st.error(f"Execution failed: {e}")
    else:
        st.info("No queries logged yet.")

# --- Ask New Question ---
user_question = st.text_input("Ask a clinical question:") 

if user_question:
    schema = load_schema()
    prompt = build_prompt(user_question, schema)
    with st.spinner("Generating SQL query with GPT..."):
        try:
            sql_query = get_sql_from_gpt(prompt)
            st.code(sql_query, language='sql')

            result_df = run_sql(sql_query)
            st.dataframe(result_df)

            log_interaction(user_question, sql_query, result_df)
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
