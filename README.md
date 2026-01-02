Glassdoor review scoring pipeline: scrape (or load) reviews, clean them, build text features, score sentiment + five goals, and emit company-level metrics and optional visuals.

## Repo map
- `reviews_scraper.py` - **only working scraper** (pydoll/CDP + Chromium).
- `data_cleaner.py` - normalizes raw CSVs and keeps mostly US-located rows.
- `extraction.py` - builds normalized text, tokens, TF-IDF matrix, and config.
- `scorer.py` - scores sentiment and 5 goal dimensions; aggregates to company-level.
- `make_viz.py` - optional figures from scored outputs.
- `config/goal_dict.json` - goal lookup; `features_exctract/` - artifacts from feature build.
- Data folders: `OG-Reviews-CSV/`, `OG_reviews-json/`, `cleaned/`, `cleaned_US/`, `out/`.

## Prerequisites
- Python 3.10+ and Chrome/Chromium installed (Chromium is required for the scraper).
- Recommended virtual env:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- Install deps (adjust as needed):
  ```bash
  pip install pandas numpy torch transformers tqdm matplotlib scikit-learn scipy unidecode spacy pydoll
  python -m spacy download en_core_web_sm
  ```

## Quick start (local)
1) (Optional) Scrape reviews with the **working scraper**:
   ```bash
   python reviews_scraper.py --url https://www.glassdoor.com/Reviews/ACME-Reviews-E9999.htm \
     --pages 5 --csv OG-Reviews-CSV/reviews_acme.csv --out OG_reviews-json/reviews_acme.json \
     --page-delay 4 --headless
   ```
   - Required: `--url` must be a Glassdoor company Reviews/Overview URL; the script auto-adds language + sort params.
   - Useful flags: `--pages` (default 3), `--csv` to also write CSV, `--page-delay` to throttle, `--chrome-binary` if Chrome is not on PATH, `--profile-dir` for a custom Chromium profile, `--headless` to hide the browser.
   - Older scrapers (`main.py`, `pydoll_client.py`) are kept for reference; use **only** `reviews_scraper.py`.

2) Clean raw CSVs:
   - Open `data_cleaner.py` and set `RAW_DIR` to where your raw `reviews_<company>.csv` files live, and `OUT_DIR` to a writable folder (e.g., `cleaned_US/`).
   - Run:
     ```bash
     python data_cleaner.py
     ```
   - Output: cleaned CSVs in `OUT_DIR` with standardized columns (`review_id,title,pros,cons,body,rating,date,job_title,company,location`), filtered non-US rows, light text normalization.

3) Build text features:
   ```bash
   python extraction.py --in "cleaned_US/reviews_*.csv" --out features_exctract
   ```
   - Produces `features_exctract/combined_reviews.parquet`, TF-IDF matrix, vocab, and `features_exctract/config.json`.

4) Score sentiment + goals:
   ```bash
   python scorer.py
   ```
   - Inputs: `features_exctract/combined_reviews.parquet`, `features_exctract/config.json`, and `config/goal_dict.json`.
   - Outputs (under `out/`):
     - `review_scores.csv` - per-review sentiment + goal scores.
     - `company_scores.csv` - aggregated company metrics.
     - `per_company/*.csv` - trimmed per-company slices.
     - `run_report.json` - run metadata and column lists.

5) (Optional) Visualize:
   ```bash
   python make_viz.py --company_csv out/company_scores.csv --review_csv out/review_scores.csv \
     --per_company_dir out/per_company --out_dir out/figures
   ```
   - Add `--show` to display figures; images are written to `out/figures/`.

## Working scraper notes (`reviews_scraper.py`)
- Uses deterministic Glassdoor pagination (`_P` / `_IP`) and parses embedded JSON caches.
- Writes JSON to `reviews.json` by default; pass `--csv` to also write UTF-8 CSV.
- Stores a Chromium profile in `chrome-profile/` (created automatically). Delete it if you want a clean session.
- Handles consent popups and sorts by recent (`sort.sortType=RD`).
- If you see repeated pages or Cloudflare/captcha, increase `--page-delay` and/or switch to a fresh profile via `--profile-dir`.

## Tips and defaults
- Files use absolute paths in some scripts; adjust `RAW_DIR`, `OUT_DIR`, and any hardcoded paths for your machine before running.
- GPU is optional; `scorer.py` auto-falls back to CPU and adapts batch size.
- Name raw files `reviews_<company>.csv` to keep outputs organized; keep everything under `out/` or `cleaned_*` to avoid path issues.
- Goal labels expected by the scorer: `physiological`, `self_protection`, `affiliation`, `status_esteem`, `family_care` (from `config/goal_dict.json`).
