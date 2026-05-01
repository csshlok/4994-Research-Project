Glassdoor review scoring and workplace-intelligence pipeline: scrape or load employee reviews, clean them, build text features, score sentiment plus five goal domains, precompute company score caches, generate RAG evidence packets, create Gemini-backed executive summaries, and serve the results through a FastAPI backend and React dashboard.

## Repo map
- `reviews_scraper.py` - supported Glassdoor scraper (pydoll/CDP + Chromium).
- `pipeline.py` - end-to-end local pipeline runner for scrape/load, clean, extract, score, and optional visualization.
- `data_cleaner.py` - normalizes raw CSVs and keeps mostly US-located rows.
- `extraction.py` - builds normalized text, tokens, TF-IDF matrix, and extraction config.
- `scorer.py` - scores sentiment and five goal dimensions; aggregates company-level metrics.
- `make_viz.py` - optional static figures from scored outputs.
- `precompute_company_scores.py` - builds company-wise score caches from `review data/`.
- `generate_topic_artifacts.py` - creates topic cluster summaries and review-to-cluster assignments.
- `build_rag_evidence.py` - joins scored reviews, topic clusters, and raw review text into model-ready RAG evidence packets.
- `generate_gemini_rag_artifacts.py` - generates cached Gemini summaries, cluster explanations, and insight text.
- `backend/` - FastAPI service for pipeline jobs, precomputed score caches, downloads, RAG artifacts, and local/hosted deployment.
- `frontend/` - React/Vite dashboard for single-company analysis, comparison mode, topic maps, and cached RAG summaries.
- `config/goal_dict.json` - five-domain fulfillment/hindrance lexicon.
- Data folders: `review data/`, `company scores/`, `features_exctract/`, `out/`, `runs/`, `server_jobs/`.

## Goal domains
The scoring model maps employee language to five domains:
- `physiological` - pay, benefits, workload, comfort, and basic stability.
- `self_protection` - fairness, safety, trust, toxicity, retaliation, and job security.
- `affiliation` - belonging, collaboration, culture, and team connection.
- `status_esteem` - recognition, growth, promotion, feedback, and advancement.
- `family_care` - flexibility, work-life support, scheduling, and care obligations.

Each review receives sentiment and goal-level fulfillment/hindrance signals. Company summaries aggregate those signals into domain scores, cluster maps, and evidence-backed explanations.

## Prerequisites
- Python 3.10+.
- Node.js 18+ for the frontend.
- Chrome/Chromium installed for the scraper and optional screenshot capture.
- Recommended Python virtual environment:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- Install Python dependencies as needed:
  ```bash
  pip install pandas numpy torch transformers tqdm matplotlib scikit-learn scipy unidecode spacy pydoll fastapi uvicorn google-genai
  python -m spacy download en_core_web_sm
  ```
- Install frontend dependencies:
  ```bash
  cd frontend
  npm install
  ```

## Environment
For Gemini-backed RAG generation and future live comparison summaries:
```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite
```

Keep `.env.render` local or configured through the hosting provider. The frontend should never receive the Gemini API key.

## Quick start: local app
Run the backend:
```bash
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Run the frontend:
```bash
cd frontend
set VITE_API_BASE_URL=http://127.0.0.1:8000&& npm run dev -- --host 127.0.0.1 --port 8080
```

Open:
```text
http://127.0.0.1:8080/
```

Useful backend checks:
```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/scored-companies
curl http://127.0.0.1:8000/api/scored-company/microsoft/rag
```

## Quick start: pipeline
1) Optional scrape with the supported scraper:
   ```bash
   python reviews_scraper.py --url https://www.glassdoor.com/Reviews/ACME-Reviews-E9999.htm ^
     --pages 5 --csv OG-Reviews-CSV/reviews_acme.csv --out OG_reviews-json/reviews_acme.json ^
     --page-delay 4 --headless
   ```

2) Clean raw CSVs:
   ```bash
   python data_cleaner.py
   ```

3) Build text features:
   ```bash
   python extraction.py --in "cleaned_US/reviews_*.csv" --out features_exctract
   ```

4) Score sentiment and goals:
   ```bash
   python scorer.py
   ```

5) Optional static visualization:
   ```bash
   python make_viz.py --company_csv out/company_scores.csv --review_csv out/review_scores.csv ^
     --per_company_dir out/per_company --out_dir out/figures
   ```

## Company score cache
The deployed dashboard uses precomputed company-wise caches so users do not wait for the full scoring pipeline on a single-CPU host.

Generate one company:
```bash
.venv\Scripts\python.exe precompute_company_scores.py --company microsoft
```

Generate all companies from `review data/`:
```bash
.venv\Scripts\python.exe precompute_company_scores.py
```

Generate topic artifacts from scored reviews:
```bash
.venv\Scripts\python.exe generate_topic_artifacts.py
```

The cache layout is:
```text
company scores/{company}/
  company_scores.csv
  review_scores.csv
  cleaned_reviews.csv
  topic_summary.csv
  topic_assignments.csv
  per_company/{company}.csv
```

## RAG artifact generation
Stage one builds evidence packets without calling a model:
```bash
.venv\Scripts\python.exe build_rag_evidence.py --company microsoft
.venv\Scripts\python.exe build_rag_evidence.py
```

Outputs:
```text
company scores/{company}/rag_evidence.json
company scores/{company}/rag_profile.json
```

Stage two calls Gemini and caches generated language:
```bash
.venv\Scripts\python.exe generate_gemini_rag_artifacts.py --company microsoft --force
.venv\Scripts\python.exe generate_gemini_rag_artifacts.py --force
```

Outputs:
```text
company scores/{company}/rag_summary.json
company scores/{company}/rag_clusters.json
company scores/{company}/rag_insights.json
```

These files power the single-company executive summary, key strengths, key risks, per-cluster descriptions, and "what stands out" section. Gemini is used during pre-cache generation, while the Render backend serves cached JSON instantly.

## Backend API surface
- `GET /api/health` - service health and queue metadata.
- `GET /api/companies` - raw review-data folders.
- `GET /api/scored-companies` - companies with precomputed score caches.
- `GET /api/scored-company/{company_id}/outputs` - list downloadable cache files.
- `GET /api/scored-company/{company_id}/download?path=...` - download cache artifacts.
- `GET /api/scored-company/{company_id}/rag` - cached Gemini/RAG summary, cluster, insight, evidence, and profile payloads.
- `POST /api/run` - queue a cache-mode pipeline job when a precomputed cache is unavailable.
- `GET /api/job/{job_id}` - job status.
- `GET /api/job/{job_id}/outputs` - generated job outputs.
- `GET /api/job/{job_id}/download?path=...` - download job artifacts.

## Frontend application
The React dashboard supports:
- Single-company analysis from the landing page.
- Company comparison from the landing page.
- Compare-from-results flow for a selected company.
- Cached executive summaries from Gemini RAG artifacts.
- Goal-domain cards, bar/radar charts, topic bubble maps, and cluster explanation cards.
- Comparison heatmap, domain gap analysis, final goal score profile, shared cluster map, and company detail view.
- Download links for cleaned reviews, review scores, aggregated scores, and topic clusters.

Cached analyses intentionally show a short loading buffer so the user sees the same processing state as full pipeline runs.

## Scraper CLI arguments (`reviews_scraper.py`)

| Long flag | Short | Description |
|---|---|---|
| `--url` | `-u` | Glassdoor company Reviews or Overview URL (required). |
| `--pages` | `-p` | Number of pages to fetch when `--end-page` is not set (default 3). |
| `--start-page` | `--start-page` | First review page number to scrape (default 1). |
| `--end-page` | `--end-page` | Last review page number to scrape (inclusive). Overrides `--pages`. |
| `--csv` | `--csv` | Optional CSV path to write alongside JSON. |
| `--out` | `-o` | JSON output path (default `reviews.json`). |
| `--page-delay` | `--page-delay` | Seconds to sleep between pages after extraction (default 3.0). |
| `--headless` | `--headless` | Run Chrome headless. |
| `--chrome-binary` | `--chrome-binary` | Path to Chrome/Chromium if not on PATH. |
| `--profile-dir` | `--profile-dir` | Custom Chromium profile directory (default `chrome-profile/`). |
| `--timeout` | `--timeout` | Overall timeout in seconds (default 1600). |

Range example:
```bash
python reviews_scraper.py --url https://www.glassdoor.com/Reviews/ACME-Reviews-E9999.htm ^
  --start-page 71 --end-page 91 --csv OG-Reviews-CSV/reviews_acme_p71_91.csv
```

## Outputs and sample frontend captures
Core scored outputs:
- `review_scores.csv` - per-review sentiment and goal signals.
- `company_scores.csv` - company-level aggregate metrics.
- `topic_summary.csv` - cluster-level language signals.
- `topic_assignments.csv` - review-to-cluster assignments.
- `rag_evidence.json` - compact evidence packets for model generation.
- `rag_profile.json` - company-level RAG profile.
- `rag_summary.json` - Gemini executive summary, strengths, risks, and domain explanations.
- `rag_clusters.json` - Gemini cluster summaries.
- `rag_insights.json` - Gemini "what stands out" observations.

Optional static figures live in `out/figures/`; radar example:

![Company radar](out/figures/06_company_radar_all.png)

Current frontend captures were generated locally and are intentionally ignored by Git:

![Frontend landing](local_compare_outputs/screenshots/frontend_landing.png)

![Microsoft analysis](local_compare_outputs/screenshots/frontend_microsoft_results.png)

![Company comparison](local_compare_outputs/screenshots/frontend_company_comparison.png)

Regenerate them by running the local app and capturing the dashboard into `local_compare_outputs/screenshots/`.

## Notes and defaults
- `company scores/` is designed for deployable cache artifacts; report/log files are not required for serving the dashboard.
- `local_compare_outputs/`, `server_jobs/`, and `runs/` are local runtime folders and should stay ignored.
- The scoring pipeline can run on CPU; precomputing company scores keeps Render deployment practical.
- Gemini generation should be cached before deployment where possible. Live Gemini should be reserved for user-selected comparisons or future employee-to-company matching flows.
- `wells_fargo` is excluded from the current score cache because it does not have the same usable review input shape as the cached companies.
