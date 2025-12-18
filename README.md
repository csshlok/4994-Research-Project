Glassdoor goal and sentiment scoring pipeline that cleans scraped review data, builds features, scores goal fulfillment/hindrance, and produces company-level metrics and visualizations.

**What this project does**
- Cleans raw Glassdoor review CSVs and standardizes columns.
- Extracts normalized text features and tokens for downstream scoring.
- Scores each review for sentiment and five goal dimensions (physiological, self_protection, affiliation, status_esteem, family_care), then aggregates to company-level metrics.
- Generates CSV outputs plus optional visualizations and per-company slices.
- Provides optional scrapers (Selenium- and pydoll-based) if you need to collect the raw reviews yourself.

**Prerequisites**
- Python 3.10+ recommended.
- Install dependencies (adjust if you already have them):
```
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy torch transformers tqdm matplotlib scikit-learn scipy unidecode spacy
python -m spacy download en_core_web_sm
```
- Chrome/Chromium installed if you plan to run the scrapers (`main.py`, `pydoll_client.py`, or `reviews_scraper.py`).

**Key inputs**
- Raw review CSVs named like `reviews_<company>.csv`. Update `RAW_DIR` and `OUT_DIR` in `data_cleaner.py` to point to your raw and cleaned paths.
- Goal dictionary: `config/goal_dict.json`.
- Feature config produced by `extraction.py`: `features_exctract/config.json`.

**Pipeline steps to reproduce**
1) Clean raw CSVs (duplicates, empty rows, light text normalization):
```
python data_cleaner.py
```
   - Adjust `RAW_DIR`/`OUT_DIR` in `data_cleaner.py` for your file locations.
   - Output: cleaned CSVs in the configured `OUT_DIR`.

2) Build features (normalized text, tokens, TF-IDF artifacts):
```
python extraction.py --in "cleaned_US/reviews_*.csv" --out features_exctract
```
   - Produces `features_exctract/combined_reviews.parquet`, TF-IDF matrix, vocab, and `config.json`.

3) Score sentiment and goals:
```
python scorer.py
```
   - Inputs: `features_exctract/combined_reviews.parquet`, `features_exctract/config.json`, `config/goal_dict.json`.
   - Outputs in `out/`:
     - `review_scores.csv`: per-review sentiment and goal scores.
     - `company_scores.csv`: aggregated company metrics (smoothed sentiment, goal means).
     - `per_company/*.csv`: trimmed per-company review slices.
     - `run_report.json`: run metadata and column lists.

4) Generate visualizations (optional):
```
python make_viz.py --company_csv out/company_scores.csv --review_csv out/review_scores.csv --per_company_dir out/per_company --out_dir out/figures
```
   - Add `--show` to display interactively. Figures are written to `out/figures/`.

5) (Optional) Scrape reviews if you do not have raw CSVs:
- Selenium path: `python main.py --url <company_overview_or_reviews_url> --username <email> --password <pwd> --limit 25` (tune args; see file for options such as `--manual_nav` and date filters).
- Pydoll/CDP path: `python reviews_scraper.py --url <reviews_url> --pages 3 --out reviews.json --csv reviews.csv`.
- Minimal pydoll example: `python pydoll_client.py` (adjust URL in `_demo()` or call from your own script).

**Outputs**
- `out/review_scores.csv`: review_id, company_id, date, n_tokens, `S_raw`, plus F_raw_*, H_raw_*, G_ratio_*, w_sent_*, G_final_* columns per goal.
- `out/company_scores.csv`: company-level counts, mean sentiment, confidence bounds, smoothed sentiment, positive/negative shares, and goal means/smoothed goal scores.
- `out/per_company/`: per-company trimmed CSVs with review-level goal scores.
- `out/figures/`: optional PNG plots from `make_viz.py`.
- `out/run_report.json`: metadata, column inventories, and parameter settings used.

**Tips**
- If GPU is unavailable, `scorer.py` falls back to CPU and auto-adjusts batch size.
- Ensure `config/goal_dict.json` matches the goal codes expected by the scorer (phys, selfprot, aff, stat, fam).
- For large runs, prefer running inside the provided virtual environment and keep outputs under `out/` to avoid path issues.
