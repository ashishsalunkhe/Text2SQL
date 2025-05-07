# 🩺 Text-to-SQL System for MIMIC-III Dataset

**Ashish Salunkhe**
University of Maryland, College Park
**Aryaman Paigankar**
University of Maryland, College Park

---
## How to reproduct this project?

This guide will help you set up and run the Text-to-SQL system for querying the MIMIC-III dataset using natural language.

---

### Clone the Repository

```bash
git clone https://github.com/your-username/mimic-llm-text2sql.git
cd mimic-llm-text2sql
```

---

### Set Up the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate     # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Download the Dataset

* Go to [Kaggle – mimic-iii-10k dataset](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k)
* Download only the `*_random.csv` files
* Place them in the following directory:

```bash
data/csv/
```

---

### Set OpenAI API Key

* Create a file named `.streamlit/secrets.toml` at the root of your repo
* Add your OpenAI key as follows:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

---

### Build the SQLite DB and Schema Map

```bash
python app/main.py --question "What are the most common diagnoses?"
```

This will create `mimic_iii.db` and `schema_map.json` in the `data/` directory.

---

### Run the Streamlit UI App

```bash
streamlit run app/ui.py
```

This will open a browser where you can ask clinical questions in plain English.

---

### Run via Command-Line (Optional)

You can also run the pipeline through CLI:

```bash
python app/main.py --question "Which lab tests are common in diabetic patients?"
```


---
## 🚀 Repo Structure

```
mimic_text_to_sql/
├── data/
│   ├── mimic_iii.db         # SQLite DB from CSVs
│   ├── schema_map.json      # JSON schema metadata
│   └── query_log.csv        # Logged questions, SQL, results
├── app/
│   ├── main.py              # CLI interface
│   ├── ui.py                # Streamlit interface
├── .streamlit/
│   └── secrets.toml         # API keys (ignored)
├── requirements.txt
└── README.md
```
---

## 📌 Problem Formulation

Large Language Models (LLMs) have shown increasing capability in natural language understanding and structured data reasoning. One practical application is translating natural language questions into SQL queries to access complex medical datasets like MIMIC-III.

Our project aims to develop a Retrieval-Augmented Generation (RAG) based Text-to-SQL system that enables healthcare professionals or researchers to interact with the MIMIC-III clinical database using plain English queries.

Key challenges addressed:

* Schema complexity
* Ambiguity in natural language
* Lack of join/contextual awareness in naive LLMs

We mitigate these issues via schema-aware metadata retrieval and GPT-based SQL generation.

---

## 🗃️ Dataset Description

We used the **mimic-III-10k** dataset — a curated subset of the full MIMIC-III clinical dataset containing \~10,000 patients.

* Source: Beth Israel Deaconess Medical Center (via PhysioNet)
* Format: 25 CSV tables (\~6 GB total)
* Relational schema includes:

  * `PATIENTS`: Demographics
  * `ADMISSIONS`: Hospital admission logs
  * `ICUSTAYS`: ICU-level data
  * `DIAGNOSES_ICD`: ICD-9 medical codes

### Data Ingestion Pipeline

* Loaded CSVs into **SQLite** for fast, structured access
* Optionally support **PostgreSQL** for scale
* Explored initial joins using `subject_id`, `hadm_id`, and `icustay_id`

---

## 📊 Descriptive Analysis

We began by understanding key patient journeys using 4 main tables:

* Explored relationships between `PATIENTS`, `ADMISSIONS`, `ICUSTAYS`, and `DIAGNOSES_ICD`
* Identified key identifiers for joins: `subject_id`, `hadm_id`
* Highlighted distribution of diagnoses and ICU visits

We also set up:

* SQLite database from CSV
* Initial EDA in Google Colab using Pandas
* Schema inspection for metadata modeling

---

## 🧠 Methodology: RAG-based LLM System

### 🔧 System Steps:

1. **Metadata Extraction**

   * Generate JSON summaries of table schemas (columns, types, join keys)
2. **Embedding Generation**

   * Use `all-MiniLM-L6-v2` from SentenceTransformers
   * Encode schema metadata into dense vectors
3. **Vector DB (ChromaDB)**

   * Store embeddings and enable semantic retrieval
4. **Retrieval Layer**

   * Given a user question, retrieve top-k relevant table schemas
5. **Prompt Construction**
g
   * Inject schema context + user query into GPT-3.5-Turbo prompt
6. **LLM SQL Generation**

   * Parse GPT output to SQL, execute, and return results

All components were orchestrated within a modular Python architecture with `main.py` (CLI) and `ui.py` (Streamlit).


---

## 📈 Evaluation Strategy

### 🎯 Ground Truth Creation

* Defined 15 clinical questions with gold SQL and results
* Example: *"What procedures are most common among deceased patients?"*
* Evaluated SQL outputs for correctness and execution success

### 📏 Metrics

| Metric                     | Description                                    |
| -------------------------- | ---------------------------------------------- |
| Execution Accuracy         | % of SQL queries that executed without error   |
| Result Overlap (Jaccard)   | Match between LLM vs. ground truth results     |
| Schema Retrieval Precision | % of correct tables retrieved in top-k context |
| Prompt Token Size          | Avg tokens used in prompt to GPT-3.5           |
| Latency / Cost             | Time + API cost per query                      |

---

## 📊 Results Summary

* ✅ Execution Accuracy: **87%** (13/15 queries successful)
* ✅ Result Overlap (Jaccard): Avg **0.72**
* ✅ Retrieval hit rate: **90%** relevant tables in top-k
* ⚠️ Common failure: SQL hallucination in JOINs or WHERE clauses

---

## ⚠️ Challenges & Takeaways

* Complex schema with repeated identifiers across tables
* Token limit requires prompt compression / top-k filtering
* LLMs occasionally hallucinate JOIN conditions
* Some vague queries required schema-specific disambiguation

---

## 🔭 Future Work

* Fine-tune with healthcare-specific SQL data (MimicSQL, Spider)
* Add error-handling and user-guided corrections
* Integrate with PostgreSQL for production-scale queries
* Explore open-source LLMs with local inference (e.g., SQLCoder)

---

## 📚 Related Work

* MimicSQL: Fine-tuned Text2SQL on MIMIC (Zhang et al., 2023)
* RAG-based Question Answering (Lewis et al., 2020)
* Spider Benchmark for cross-domain SQL generation (Yu et al., 2018)

---

## 🔗 References

* Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database. [https://doi.org/10.13026/C2XW26](https://doi.org/10.13026/C2XW26)
* mimic-III-10k \[Kaggle]. [https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k)
* Lewis, P. et al. (2020). Retrieval-Augmented Generation. NeurIPS 33
* Yu, T. et al. (2018). Spider Dataset. EMNLP
* Zhang, H. et al. (2023). MimicSQL. ACL. [https://arxiv.org/abs/2305.11921](https://arxiv.org/abs/2305.11921)

---

## 👥 Authors

* **Ashish Salunkhe** — [ashishsalunke.com](https://ashishsalunke.com)
* **Aryaman Paigankar** — University of Maryland


