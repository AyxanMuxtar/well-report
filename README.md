# eiLink RA-DS 2026 — Drilling Report Pipeline

End-to-end pipeline for processing drilling daily report PDFs:

1. **Task 1** — Parse PDFs and store structured data in DuckDB
2. **Task 2** — Match NDS events to operations using TF-IDF + sentence-transformers
3. **Task 3** — NLP analysis (NER, activity classification, TF-IDF keywords)

---

## How this project is organized

This is a **hybrid notebook + library** project:

- **`src/`** holds reusable code as importable Python modules (parsers, DB helpers, matchers).
  Edit these when you need to fix logic. Keeps the GitHub deliverable clean.
- **`notebooks/`** holds runnable Jupyter notebooks that orchestrate work and visualize results.
  This is where you actually *run things* and *see output*. Each notebook imports from `src/`.

```
eilink_ra_ds/
├── data/
│   ├── raw_pdfs/                    # Drop all 1,390 PDFs here
│   ├── nds_events.xlsx              # Drop the NDS events spreadsheet here
│   └── processed/                   # Cached indexes (TF-IDF, SBERT embeddings)
├── db/
│   └── drilling.duckdb              # DuckDB database (created by Task 1)
│
├── src/                             ← REUSABLE LIBRARY CODE
│   ├── common/                      # Shared: config, db, logging, wellbore helpers
│   ├── ingestion/                   # Task 1: PDF parsing → DuckDB
│   ├── task2_matching/              # Task 2: NDS event matching
│   └── task3_nlp/                   # Task 3: NER, activity tags, keywords
│
├── notebooks/                       ← WHERE YOU WORK
│   ├── 00_explore.ipynb             # Free-form scratch / inspection
│   ├── 01_task1_ingest.ipynb        # Run ingestion, inspect DB
│   ├── 02_task2_matching.ipynb      # NDS event matching
│   └── 03_task3_nlp.ipynb           # NLP analysis
│
├── outputs/                         # CSVs, reports, deliverables
├── requirements.txt
└── README.md
```

---

## Setup (Windows + NVIDIA GPU)

### 1. Create environment
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install dependencies (GPU-accelerated PyTorch first)
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Place data
- Copy all PDFs into `data/raw_pdfs/`
- Copy `nds_events.xlsx` into `data/`

### 4. Launch Jupyter
```powershell
jupyter lab        # or:  jupyter notebook
```

---

## Usage — run notebooks in order

### `01_task1_ingest.ipynb` → Task 1
1. Smoke-tests the parser on one PDF (so you can spot bugs before full run)
2. Runs full ingestion across all 1,390 PDFs (about 5 to 15 minutes)
3. Inspects the database with sample queries
4. Plots reports per well over time

After this notebook completes, `db/drilling.duckdb` is populated with around 10 to 14 thousand operation rows.

### `02_task2_matching.ipynb` → Task 2 
Loads `nds_events.xlsx`, builds TF-IDF + SBERT indexes, runs strict matching for F-10/F-11/F-12 and aggressive cross-well fallback for F-13.

### `03_task3_nlp.ipynb` → Task 3 
Runs NER, activity classification, and TF-IDF keyword extraction over the operations table.

### `00_explore.ipynb` → Scratch
Use this whenever you want to poke at a specific PDF, run an ad-hoc query, or test an idea before adding it to one of the main notebooks.

---

## Notebook tip: auto-reload

All notebooks include this near the top:
```python
%load_ext autoreload
%autoreload 2
```

This means: edit a `.py` file in `src/`, save, and just rerun the affected notebook cell. No kernel restart needed. Critical for fast iteration.

---

## Outputs

| File | Description |
|---|---|
| `db/drilling.duckdb` | All structured data (reports, operations, fluid, etc.) |
| `outputs/task2_matches.csv` | NDS event → operation matches (TF-IDF + SBERT, top-K) |
| `outputs/task2_benchmark.csv` | Method comparison for Task 2 |
| `outputs/task3_entities.csv` | Extracted NER entities |
| `outputs/task3_activity_tags.csv` | Normalized activity labels per operation |
| `outputs/task3_keywords.csv` | Top-N TF-IDF keywords per report |
| `outputs/frequent_events_per_well.csv` | Bonus: frequent problem patterns per well |

---

## Author

Ayxan Muxtar
