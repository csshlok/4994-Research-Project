from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from csv_safety import sanitize_csv_row

try:
    import pandas as pd
except Exception:
    pd = None


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def safe_slug(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text).strip())
    s = s.strip("_").lower()
    return s or "job"


def _company_match_key(text: str) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def overview_to_reviews_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    if not parsed.scheme or not parsed.netloc:
        return url
    if "glassdoor." not in parsed.netloc.lower():
        return url

    path = parsed.path or ""
    if "/Reviews/" in path:
        return url
    if "/Overview/" not in path:
        return url

    m = re.search(r"Working-at-([^-]+)-EI_IE(\d+)", path, flags=re.I)
    if not m:
        return url

    company_slug = m.group(1)
    employer_id = m.group(2)
    new_path = f"/Reviews/{company_slug}-Reviews-EI_IE{employer_id}.htm"
    return urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", ""))


def _normalize_region(region: str | None) -> str | None:
    if not region:
        return None
    cleaned = re.sub(r"\s+", " ", str(region).strip())
    return cleaned or None


def _apply_region_filters(qd: Dict[str, List[str]], region: str | None) -> None:
    reg = _normalize_region(region)
    if not reg:
        return

    qd["filter.location"] = [reg]

    us_aliases = {"united states", "united states of america", "us", "u.s.", "usa", "u.s.a."}
    if reg.casefold() in us_aliases:
        qd["filter.locationId"] = ["1"]
        qd["filter.locationType"] = ["N"]
    else:
        # Prevent stale IDs from conflicting with a non-US region string.
        qd.pop("filter.locationId", None)
        qd.pop("filter.locationType", None)


def canonicalize_reviews_url(url: str, region: str | None = None) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if not parsed.scheme or not parsed.netloc:
        return url

    path = parsed.path or ""
    if not re.search(r"/reviews", path, flags=re.I) and not re.search(r"-reviews-e\d+", path, flags=re.I):
        return url

    path_base = re.sub(r"(_P\d+|_IP\d+)?\.htm.*$", "", path, flags=re.I)
    path_final = f"{path_base}.htm"

    qd: Dict[str, List[str]] = {}
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        qd.setdefault(k, []).append(v)

    qd.setdefault("filter.iso3Language", ["eng"])
    qd.setdefault("sort.sortType", ["RD"])
    qd.setdefault("sort.ascending", ["false"])
    _apply_region_filters(qd, region)

    query = urlencode(qd, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, path_final, "", query, ""))


def resolve_add_company_dir(add_root: Path, job_name: str) -> Path:
    add_root = add_root.resolve()
    add_root.mkdir(parents=True, exist_ok=True)

    wanted = str(job_name or "").strip()
    if not wanted:
        raise ValueError("--job must be non-empty when using --add.")

    dirs = [p for p in add_root.iterdir() if p.is_dir()]
    for p in dirs:
        if p.name == wanted:
            return p

    wanted_lower = wanted.lower()
    for p in dirs:
        if p.name.lower() == wanted_lower:
            return p

    wanted_slug = safe_slug(wanted)
    for p in dirs:
        if safe_slug(p.name) == wanted_slug:
            return p

    wanted_key = _company_match_key(wanted)
    for p in dirs:
        if _company_match_key(p.name) == wanted_key:
            return p

    # No match found, create a new folder with the requested name.
    new_dir = add_root / wanted
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(x) for x in (reader.fieldnames or []) if x]
        rows: list[dict[str, str]] = []
        for row in reader:
            if not isinstance(row, dict):
                continue
            rec: dict[str, str] = {}
            for k, v in row.items():
                if k is None:
                    continue
                key = str(k)
                if not key:
                    continue
                rec[key] = "" if v is None else str(v)
            if rec:
                rows.append(rec)
    return rows, fields


def _row_identity_key(row: dict[str, str]) -> tuple[str, str]:
    rid = str(row.get("review_id", "")).strip()
    if rid:
        return ("review_id", rid)

    alt_id = str(row.get("id", "")).strip()
    if alt_id:
        return ("id", alt_id)

    parts = [
        str(row.get("title", "")).strip().lower(),
        str(row.get("date", "")).strip().lower(),
        str(row.get("pros", "")).strip().lower(),
        str(row.get("cons", "")).strip().lower(),
        str(row.get("body", "")).strip().lower(),
    ]
    joined = "|".join(parts).strip("|")
    if joined:
        return ("text_sig", joined)

    # Last-resort stable fingerprint from sorted JSON fields.
    return ("json", json.dumps(row, sort_keys=True, ensure_ascii=False))


def _fill_missing(dst: dict[str, str], src: dict[str, str]) -> bool:
    changed = False
    for k, v in src.items():
        vtxt = str(v).strip()
        if not vtxt:
            continue
        if not str(dst.get(k, "")).strip():
            dst[k] = str(v)
            changed = True
    return changed


def merge_scraped_reviews_into_cache(new_csv: Path, cache_csv: Path) -> dict[str, Any]:
    new_rows, new_fields = _read_csv_rows(new_csv)
    if not new_rows:
        raise ValueError(f"No rows found in scraped CSV: {new_csv}")

    old_rows, old_fields = _read_csv_rows(cache_csv)

    field_order: list[str] = []
    for f in old_fields + new_fields:
        if f and f not in field_order:
            field_order.append(f)
    if not field_order:
        raise ValueError("Cannot determine CSV header fields for merge.")

    merged: list[dict[str, str]] = []
    index: dict[tuple[str, str], int] = {}

    for row in old_rows:
        key = _row_identity_key(row)
        if key in index:
            _fill_missing(merged[index[key]], row)
            continue
        index[key] = len(merged)
        merged.append(dict(row))

    added = 0
    updated = 0
    for row in new_rows:
        key = _row_identity_key(row)
        if key in index:
            if _fill_missing(merged[index[key]], row):
                updated += 1
            continue
        index[key] = len(merged)
        merged.append(dict(row))
        added += 1

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    with cache_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in merged:
            rec = {k: row.get(k, "") for k in field_order}
            writer.writerow(sanitize_csv_row(rec))

    return {
        "cache_csv": str(cache_csv),
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "added_rows": added,
        "updated_rows": updated,
        "final_rows": len(merged),
    }


class PipelineLogger:
    def __init__(self, log_path: Path, progress_path: Path):
        self.log_path = log_path
        self.progress_path = progress_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{now_ts()}] {msg}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def progress(self, stage: str, status: str, **fields: Any) -> None:
        rec = {"ts": now_ts(), "stage": stage, "status": status}
        rec.update(fields)
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def run_command(cmd: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command:\n  " + " ".join(cmd) + "\n\n")
        res = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    return res.returncode


def wait_for_pause_trigger(log_path: Path, proc: subprocess.Popen,
                           timeout_sec: float = 60.0) -> str | None:
    start = time.time()
    while (time.time() - start) < timeout_sec:
        if proc.poll() is not None:
            return None
        try:
            if log_path.exists():
                data = log_path.read_text(encoding="utf-8", errors="ignore")
                page_idx = data.find("[page] 1/")
                if page_idx != -1:
                    landed_idx = data.find("[nav] Landed:", page_idx)
                    if landed_idx != -1:
                        return "first_page"
                if "[challenge] Detected" in data:
                    return "challenge"
                if "[nav] Base resolved:" in data:
                    return "base_resolved"
        except Exception:
            pass
        time.sleep(0.2)
    return None


def _nt_suspend_resume(pid: int, suspend: bool) -> tuple[bool, str | None]:
    if platform.system().lower() != "windows":
        return False, "unsupported OS"
    PROCESS_SUSPEND_RESUME = 0x0800
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not handle:
        err = ctypes.windll.kernel32.GetLastError()
        return False, f"OpenProcess failed: {err}"
    try:
        func = ctypes.windll.ntdll.NtSuspendProcess if suspend else ctypes.windll.ntdll.NtResumeProcess
        status = func(handle)
        if status != 0:
            return False, f"NtSuspendResume failed: 0x{int(status):08X}"
        return True, None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def suspend_process(pid: int) -> tuple[bool, str | None]:
    return _nt_suspend_resume(pid, True)


def resume_process(pid: int) -> tuple[bool, str | None]:
    return _nt_suspend_resume(pid, False)


def pause_for_user(message: str) -> None:
    try:
        input(message)
    except EOFError:
        pass


def run_scraper_command(cmd: List[str], log_path: Path, cwd: Path,
                        pause_seconds: float, pause_until_enter: bool,
                        logger: "PipelineLogger") -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command:\n  " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)

        if pause_until_enter or (pause_seconds and pause_seconds > 0):
            trigger = wait_for_pause_trigger(log_path, proc, timeout_sec=60.0)
            if trigger and proc.poll() is None:
                ok, err = suspend_process(proc.pid)
                reason = {
                    "first_page": "first page load",
                    "challenge": "challenge detected",
                    "base_resolved": "base resolved",
                }.get(trigger, "pause trigger")

                if ok:
                    if pause_until_enter:
                        logger.log(f"[scrape] Paused after {reason}; waiting for Enter.")
                        pause_for_user("Solve any captcha/login, then press Enter to continue...")
                    if pause_seconds and pause_seconds > 0:
                        logger.log(f"[scrape] Pausing {pause_seconds:.1f}s after {reason}.")
                        time.sleep(float(pause_seconds))
                    ok2, err2 = resume_process(proc.pid)
                    if not ok2:
                        logger.log(f"[scrape] Resume failed; scraper may still be paused. {err2}")
                else:
                    logger.log(f"[scrape] Suspend failed/unsupported; pausing without suspend after {reason}. {err}")
                    if pause_until_enter:
                        pause_for_user("Solve any captcha/login, then press Enter to continue...")
                    if pause_seconds and pause_seconds > 0:
                        time.sleep(float(pause_seconds))
            else:
                logger.log("[scrape] Pause skipped (trigger not detected or process ended).")

        return proc.wait()


def resolve_files(arg: str, dir_pattern: str = "reviews_*.csv") -> List[Path]:
    p = Path(arg)
    if p.exists():
        if p.is_file():
            return [p]
        if p.is_dir():
            matches = sorted(p.glob(dir_pattern))
            if not matches:
                matches = sorted(p.glob("*.csv"))
            return matches
    return [Path(x) for x in glob.glob(arg)]


def copy_files(files: List[Path], dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for fp in files:
        dst = dest_dir / fp.name
        shutil.copy2(fp, dst)
        copied.append(dst)
    return copied


def copy_dir(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing directory: {src_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


def list_files_rel(root: Path, base: Path) -> List[str]:
    if not base.exists():
        return []
    files = [p for p in base.rglob("*") if p.is_file()]
    return [str(p.relative_to(root)) for p in files]


def count_csv_rows(path: Path) -> int | None:
    if pd is None or not path.exists():
        return None
    try:
        return int(pd.read_csv(path).shape[0])
    except Exception:
        return None


def count_parquet_rows(path: Path) -> int | None:
    if pd is None or not path.exists():
        return None
    try:
        return int(pd.read_parquet(path).shape[0])
    except Exception:
        return None


def read_columns(path: Path) -> List[str] | None:
    if pd is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=5)
        return list(df.columns)
    except Exception:
        return None


def write_env(meta_dir: Path, logger: PipelineLogger) -> None:
    env_path = meta_dir / "env.txt"
    lines = [
        f"python={platform.python_version()}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"cwd={Path.cwd()}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pip_path = meta_dir / "pip_freeze.txt"
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            pip_path.write_text(res.stdout, encoding="utf-8")
        else:
            logger.log(f"[meta] pip freeze failed: rc={res.returncode}")
    except Exception as e:
        logger.log(f"[meta] pip freeze error: {e}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run end-to-end pipeline (scrape -> clean -> extract -> score -> viz).")

    ap.add_argument("--job", required=True, help="Job/company label (e.g., Amazon).")
    ap.add_argument("--url", default=None, help="Glassdoor Reviews/Overview URL.")
    ap.add_argument(
        "--region",
        default=None,
        help=(
            "Optional Glassdoor location filter, for example 'United States'. "
            "For United States, filter.locationId=1 and filter.locationType=N are enforced."
        ),
    )

    ap.add_argument("--run-root", default="runs", help="Top-level folder for runs.")
    ap.add_argument("--run-id", default=None, help="Run ID (default: YYYYMMDD_HHMMSS).")
    ap.add_argument("--keep-intermediate", nargs="?", const=True, default=True, type=parse_bool,
                    help="Keep intermediate artifacts (default: true).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing run folder.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue after stage failures.")

    ap.add_argument("--skip-scrape", action="store_true")
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--scrape-only", action="store_true",
                    help="Run only scrape stage, then skip clean/extract/score/viz without requiring precomputed inputs.")
    ap.add_argument("--add", action="store_true",
                    help="After scraping, merge new reviews into <add-root>/<job>/reviews.csv (deduped).")
    ap.add_argument("--add-root", default="review data",
                    help="Root folder for --add merge target (default: review data).")

    ap.add_argument("--raw-csv", default=None, help="Raw CSV path or glob (if skipping scrape).")
    ap.add_argument("--clean-glob", default=None, help="Cleaned CSV glob (if skipping clean).")
    ap.add_argument("--features-dir", default=None, help="Features dir (if skipping extract).")
    ap.add_argument("--scored-out-dir", default=None, help="Scoring output dir (if skipping score).")
    ap.add_argument("--goal-dict", default=None, help="Goal dictionary path (optional).")

    ap.add_argument("--pages", type=int, default=3,
                    help="Number of review pages to collect (used when --end-page is not set).")
    ap.add_argument("--start-page", type=int, default=1,
                    help="First review page number to scrape (default: 1).")
    ap.add_argument("--end-page", type=int, default=None,
                    help="Last review page number to scrape (inclusive). If set, it overrides --pages count.")
    ap.add_argument("--page-delay", type=float, default=3.0, help="Seconds to wait between pages.")
    ap.add_argument("--headless", action="store_true", help="Run Chrome headless.")
    ap.add_argument("--pause-seconds", type=float, default=0.0,
                    help="Seconds to wait after the first page loads (or on challenge).")
    ap.add_argument("--pause-until-enter", action="store_true",
                    help="Pause after the first page loads (or on challenge) until Enter is pressed.")
    ap.add_argument("--chrome-binary", default=None, help="Path to Chrome binary.")
    ap.add_argument("--profile-dir", default=None, help="Chrome profile dir (default: chrome-profile).")
    ap.add_argument("--timeout", type=int, default=1600, help="Overall scraper timeout (seconds).")

    # NEW: forward challenge behavior into reviews_scraper.py
    ap.add_argument("--scrape-challenge-mode", choices=["block", "log_only"], default="log_only",
                    help="Forward to reviews_scraper.py --challenge-mode (block|log_only).")
    ap.add_argument("--scrape-pause-until-enter", action="store_true",
                    help="Forward pause-until-enter to reviews_scraper.py (block mode).")
    ap.add_argument("--scrape-challenge-wait", type=float, default=0.0,
                    help="Forward challenge-wait seconds to reviews_scraper.py (block mode).")
    ap.add_argument("--scrape-challenge-max-retries", type=int, default=3,
                    help="Forward max retries per page to reviews_scraper.py (block mode).")
    ap.add_argument("--scrape-challenge-retry-backoff", type=float, default=3.0,
                    help="Forward retry backoff multiplier to reviews_scraper.py (block mode).")
    ap.add_argument("--scrape-hydrate-timeout", type=float, default=12.0,
                    help="Forward hydrate timeout to reviews_scraper.py (seconds).")
    ap.add_argument("--scrape-stop-on-empty-pages", type=int, default=2,
                    help="Forward stop-on-empty-pages to reviews_scraper.py (default: 2).")


    return ap.parse_args()


def finalize_report(run_dir: Path, meta_dir: Path,
                    stage_results: Dict[str, Dict[str, Any]],
                    errors: List[str]) -> None:
    report = {"run_dir": str(run_dir), "stages": stage_results, "errors": errors}

    args_path = meta_dir / "pipeline_args.json"
    if args_path.exists():
        try:
            report["pipeline_args"] = json.loads(args_path.read_text(encoding="utf-8"))
        except Exception:
            report["pipeline_args"] = None

    scrape_csv = run_dir / "01_scrape" / "reviews.csv"
    clean_files = sorted((run_dir / "02_clean").glob("*.csv"))
    extract_parquet = run_dir / "03_extract" / "combined_reviews.parquet"
    review_scores = run_dir / "04_score" / "review_scores.csv"
    company_scores = run_dir / "04_score" / "company_scores.csv"
    fig_dir = run_dir / "05_viz" / "figures"

    clean_rows_total = None
    if clean_files and pd is not None:
        clean_rows_total = sum([count_csv_rows(p) or 0 for p in clean_files])

    report["counts"] = {
        "scrape_reviews_csv": count_csv_rows(scrape_csv),
        "clean_files": len(clean_files),
        "clean_rows_total": clean_rows_total,
        "extract_reviews": count_parquet_rows(extract_parquet),
        "review_scores_rows": count_csv_rows(review_scores),
        "company_scores_rows": count_csv_rows(company_scores),
        "viz_figures": len(list(fig_dir.glob("*.png"))) if fig_dir.exists() else 0,
    }

    report["columns"] = {
        "review_scores": read_columns(review_scores),
        "company_scores": read_columns(company_scores),
    }

    scorer_report = run_dir / "04_score" / "run_report.json"
    if scorer_report.exists():
        try:
            report["scorer_report"] = json.loads(scorer_report.read_text(encoding="utf-8"))
        except Exception:
            report["scorer_report"] = None

    (meta_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.scrape_only:
        args.skip_clean = True
        args.skip_extract = True
        args.skip_score = True
        args.skip_viz = True

    if args.pages < 1:
        raise ValueError("--pages must be >= 1.")
    if args.start_page < 1:
        raise ValueError("--start-page must be >= 1.")
    if args.end_page is not None and args.end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page.")
    if args.add and args.skip_scrape:
        raise ValueError("--add requires scraping; do not use it with --skip-scrape.")

    if not args.skip_scrape and not args.url:
        raise ValueError("--url is required unless --skip-scrape is set.")
    if args.skip_scrape and not args.raw_csv:
        raise ValueError("--raw-csv is required when --skip-scrape is set.")
    if args.skip_clean and not args.clean_glob and not args.scrape_only:
        raise ValueError("--clean-glob is required when --skip-clean is set.")
    if args.skip_extract and not args.features_dir and not args.scrape_only:
        raise ValueError("--features-dir is required when --skip-extract is set.")
    if args.skip_score and not args.scored_out_dir and not args.scrape_only:
        raise ValueError("--scored-out-dir is required when --skip-score is set.")

    job_slug = safe_slug(args.job)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root)
    run_dir = run_root / job_slug / run_id

    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run folder exists: {run_dir}")
        shutil.rmtree(run_dir)

    meta_dir = run_dir / "00_meta"
    scrape_dir = run_dir / "01_scrape"
    clean_dir = run_dir / "02_clean"
    extract_dir = run_dir / "03_extract"
    score_dir = run_dir / "04_score"
    viz_dir = run_dir / "05_viz"
    logs_dir = run_dir / "99_logs"

    for d in [meta_dir, scrape_dir, clean_dir, extract_dir, score_dir, viz_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger(logs_dir / "pipeline.log", logs_dir / "progress.jsonl")
    logger.log(f"[init] job={args.job} job_slug={job_slug} run_id={run_id}")

    repo_root = Path(__file__).resolve().parent
    resolved_profile_dir = args.profile_dir or str(repo_root / "chrome-profile")

    base_url = overview_to_reviews_url(args.url) if args.url else None
    resolved_url = canonicalize_reviews_url(base_url, region=args.region) if base_url else None

    args_path = meta_dir / "pipeline_args.json"
    args_payload = vars(args).copy()
    args_payload["job_slug"] = job_slug
    args_payload["run_id"] = run_id
    args_payload["run_dir"] = str(run_dir)
    args_payload["profile_dir_resolved"] = resolved_profile_dir
    if base_url and args.url and base_url != args.url:
        args_payload["url_mapped_overview_to_reviews"] = base_url
    if resolved_url:
        args_payload["url_resolved"] = resolved_url
    args_path.write_text(json.dumps(args_payload, indent=2), encoding="utf-8")
    write_env(meta_dir, logger)

    stage_results: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    def record_stage(name: str, status: str, start_ts: str | None, end_ts: str | None,
                     duration_sec: float | None, outputs: List[str], log_path: Path | None,
                     extra: Dict[str, Any] | None = None) -> None:
        rec = {
            "status": status,
            "started_at": start_ts,
            "ended_at": end_ts,
            "duration_sec": duration_sec,
            "outputs": outputs,
            "log": str(log_path) if log_path else None,
        }
        if extra:
            rec.update(extra)
        stage_results[name] = rec

    # ---------------- Scrape ----------------
    try:
        if args.skip_scrape:
            raw_files = resolve_files(args.raw_csv)
            if not raw_files:
                raise FileNotFoundError(f"No files matched: {args.raw_csv}")
            if len(raw_files) == 1:
                dst = scrape_dir / "reviews.csv"
                shutil.copy2(raw_files[0], dst)
            else:
                copy_files(raw_files, scrape_dir)
            outputs = list_files_rel(run_dir, scrape_dir)
            log_path = scrape_dir / "scrape.log"
            log_path.write_text("scrape skipped\ninputs:\n" + "\n".join([str(p) for p in raw_files]) + "\n", encoding="utf-8")
            logger.progress("scrape", "skipped", reason="user_skip", outputs=outputs)
            record_stage("scrape", "skipped", None, None, None, outputs, log_path,
                         extra={"input_sources": [str(p) for p in raw_files]})
        else:
            logger.progress("scrape", "started")
            start_ts = now_ts()
            start_time = time.time()

            reviews_json = scrape_dir / "reviews.json"
            reviews_csv = scrape_dir / "reviews.csv"
            profile_dir = resolved_profile_dir
            logger.log(f"[scrape] chrome profile dir: {profile_dir}")

            url_for_scrape = resolved_url or args.url
            if resolved_url and args.url and resolved_url != args.url:
                logger.log(f"[scrape] Canonical URL: {resolved_url}")

            cmd = [
                sys.executable, "reviews_scraper.py",
                "--url", url_for_scrape,
                "--pages", str(args.pages),
                "--start-page", str(args.start_page),
                "--page-delay", str(args.page_delay),
                "--timeout", str(args.timeout),
                "--out", str(reviews_json),
                "--csv", str(reviews_csv),
                "--challenge-mode", args.scrape_challenge_mode,
                "--hydrate-timeout", str(args.scrape_hydrate_timeout),
                "--stop-on-empty-pages", str(args.scrape_stop_on_empty_pages),

            ]
            if args.end_page is not None:
                cmd += ["--end-page", str(args.end_page)]
            if args.region:
                cmd += ["--region", str(args.region)]
            if args.headless:
                cmd.append("--headless")
            if args.chrome_binary:
                cmd += ["--chrome-binary", args.chrome_binary]
            cmd += ["--profile-dir", profile_dir]

            # block-mode extras only
            if args.scrape_pause_until_enter:
                cmd.append("--pause-until-enter")
            if args.scrape_challenge_wait and args.scrape_challenge_wait > 0:
                cmd += ["--challenge-wait", str(args.scrape_challenge_wait)]
            cmd += [
                "--challenge-max-retries", str(args.scrape_challenge_max_retries),
                "--challenge-retry-backoff", str(args.scrape_challenge_retry_backoff),
            ]

            log_path = scrape_dir / "scrape.log"
            rc = run_scraper_command(cmd, log_path, repo_root, args.pause_seconds, args.pause_until_enter, logger)

            end_ts = now_ts()
            duration = time.time() - start_time
            outputs = list_files_rel(run_dir, scrape_dir)

            if rc != 0:
                logger.progress("scrape", "failed", returncode=rc, log=str(log_path))
                record_stage("scrape", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"scrape failed (rc={rc})")

            add_merge_info: Dict[str, Any] | None = None
            if args.add:
                add_root = Path(args.add_root)
                company_dir = resolve_add_company_dir(add_root, args.job)
                cache_csv = company_dir / "reviews.csv"
                add_merge_info = merge_scraped_reviews_into_cache(reviews_csv, cache_csv)
                logger.log(
                    "[scrape][add] merged into cache: "
                    f"old={add_merge_info['old_rows']} new={add_merge_info['new_rows']} "
                    f"added={add_merge_info['added_rows']} updated={add_merge_info['updated_rows']} "
                    f"final={add_merge_info['final_rows']} file={add_merge_info['cache_csv']}"
                )

            logger.progress("scrape", "completed", outputs=outputs, add_merge=add_merge_info)
            record_stage(
                "scrape",
                "completed",
                start_ts,
                end_ts,
                duration,
                outputs,
                log_path,
                extra={"add_merge": add_merge_info} if add_merge_info else None,
            )
    except Exception as e:
        errors.append(f"scrape: {e}")
        logger.log(f"[scrape][error] {e}")
        finalize_report(run_dir, meta_dir, stage_results, errors)
        return 1

    def stage_fail_or_continue(stage_name: str, err: Exception) -> bool:
        """
        Returns True if we should continue, False if we should abort.
        """
        errors.append(f"{stage_name}: {err}")
        logger.log(f"[{stage_name}][error] {err}")
        if args.continue_on_error:
            logger.log(f"[{stage_name}] continue-on-error enabled; proceeding to next stage.")
            return True
        finalize_report(run_dir, meta_dir, stage_results, errors)
        return False

    def maybe_cleanup_stage(dir_path: Path) -> None:
        """
        If keep-intermediate is false, remove intermediate dirs after success.
        We keep 00_meta, 99_logs, 04_score, 05_viz by default because they are final outputs.
        """
        if args.keep_intermediate:
            return
        if not dir_path.exists():
            return
        shutil.rmtree(dir_path, ignore_errors=True)

    # ---------------- Clean ----------------
    try:
        if args.skip_clean:
            log_path = clean_dir / "clean.log"
            if args.scrape_only:
                log_path.write_text("clean skipped (scrape-only mode)\n", encoding="utf-8")
                outputs = list_files_rel(run_dir, clean_dir)
                logger.progress("clean", "skipped", reason="scrape_only", outputs=outputs)
                record_stage("clean", "skipped", None, None, None, outputs, log_path,
                             extra={"reason": "scrape_only"})
            else:
                src_files = resolve_files(args.clean_glob)
                if not src_files:
                    raise FileNotFoundError(f"No files matched clean-glob: {args.clean_glob}")
                copy_files(src_files, clean_dir)
                log_path.write_text("clean skipped\ninputs:\n" + "\n".join([str(p) for p in src_files]) + "\n", encoding="utf-8")
                outputs = list_files_rel(run_dir, clean_dir)
                logger.progress("clean", "skipped", reason="user_skip", outputs=outputs)
                record_stage("clean", "skipped", None, None, None, outputs, log_path,
                             extra={"input_sources": [str(p) for p in src_files]})
        else:
            logger.progress("clean", "started")
            start_ts = now_ts()
            start_time = time.time()

            cmd = [
                sys.executable, "data_cleaner.py",
                "--raw-dir", str(scrape_dir),
                "--pattern", str(scrape_dir / "reviews.csv"),
                "--out-dir", str(clean_dir),
                "--job", safe_slug(args.job),
                "--report-json", str(clean_dir / "clean_report.json"),
            ]
            log_path = clean_dir / "clean.log"
            rc = run_command(cmd, log_path, repo_root)

            end_ts = now_ts()
            duration = time.time() - start_time
            outputs = list_files_rel(run_dir, clean_dir)

            if rc != 0:
                logger.progress("clean", "failed", returncode=rc, log=str(log_path))
                record_stage("clean", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"clean failed (rc={rc})")

            logger.progress("clean", "completed", outputs=outputs)
            record_stage("clean", "completed", start_ts, end_ts, duration, outputs, log_path)
    except Exception as e:
        if not stage_fail_or_continue("clean", e):
            return 1

    # ---------------- Extract ----------------
    try:
        if args.skip_extract:
            log_path = extract_dir / "extract.log"
            if args.scrape_only:
                log_path.write_text("extract skipped (scrape-only mode)\n", encoding="utf-8")
                outputs = list_files_rel(run_dir, extract_dir)
                logger.progress("extract", "skipped", reason="scrape_only", outputs=outputs)
                record_stage("extract", "skipped", None, None, None, outputs, log_path,
                             extra={"reason": "scrape_only"})
            else:
                src_dir = Path(args.features_dir)
                copy_dir(src_dir, extract_dir)
                log_path.write_text(f"extract skipped\ninputs_dir:\n{src_dir}\n", encoding="utf-8")
                outputs = list_files_rel(run_dir, extract_dir)
                logger.progress("extract", "skipped", reason="user_skip", outputs=outputs)
                record_stage("extract", "skipped", None, None, None, outputs, log_path,
                             extra={"input_dir": str(src_dir)})
        else:
            logger.progress("extract", "started")
            start_ts = now_ts()
            start_time = time.time()

            cmd = [
                sys.executable, "extraction.py",
                "--in", str(clean_dir / "*.csv"),
                "--out", str(extract_dir),
            ]
            log_path = extract_dir / "extract.log"
            rc = run_command(cmd, log_path, repo_root)

            end_ts = now_ts()
            duration = time.time() - start_time
            outputs = list_files_rel(run_dir, extract_dir)

            if rc != 0:
                logger.progress("extract", "failed", returncode=rc, log=str(log_path))
                record_stage("extract", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"extract failed (rc={rc})")

            logger.progress("extract", "completed", outputs=outputs)
            record_stage("extract", "completed", start_ts, end_ts, duration, outputs, log_path)
    except Exception as e:
        if not stage_fail_or_continue("extract", e):
            return 1

    # ---------------- Score ----------------
    try:
        if args.skip_score:
            log_path = score_dir / "score.log"
            if args.scrape_only:
                log_path.write_text("score skipped (scrape-only mode)\n", encoding="utf-8")
                outputs = list_files_rel(run_dir, score_dir)
                logger.progress("score", "skipped", reason="scrape_only", outputs=outputs)
                record_stage("score", "skipped", None, None, None, outputs, log_path,
                             extra={"reason": "scrape_only"})
            else:
                src_dir = Path(args.scored_out_dir)
                copy_dir(src_dir, score_dir)
                log_path.write_text(f"score skipped\ninputs_dir:\n{src_dir}\n", encoding="utf-8")
                outputs = list_files_rel(run_dir, score_dir)
                logger.progress("score", "skipped", reason="user_skip", outputs=outputs)
                record_stage("score", "skipped", None, None, None, outputs, log_path,
                             extra={"input_dir": str(src_dir)})
        else:
            logger.progress("score", "started")
            start_ts = now_ts()
            start_time = time.time()

            cmd = [
                sys.executable, "scorer.py",
                "--features-dir", str(extract_dir),
                "--out-dir", str(score_dir),
            ]
            if args.goal_dict:
                cmd += ["--goal-dict", str(args.goal_dict)]

            log_path = score_dir / "score.log"
            rc = run_command(cmd, log_path, repo_root)

            end_ts = now_ts()
            duration = time.time() - start_time
            outputs = list_files_rel(run_dir, score_dir)

            if rc != 0:
                logger.progress("score", "failed", returncode=rc, log=str(log_path))
                record_stage("score", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"score failed (rc={rc})")

            logger.progress("score", "completed", outputs=outputs)
            record_stage("score", "completed", start_ts, end_ts, duration, outputs, log_path)
    except Exception as e:
        if not stage_fail_or_continue("score", e):
            return 1

    # ---------------- Viz ----------------
    try:
        if args.skip_viz:
            log_path = viz_dir / "viz.log"
            log_path.write_text("viz skipped\n", encoding="utf-8")
            outputs = list_files_rel(run_dir, viz_dir)
            logger.progress("viz", "skipped", reason="user_skip", outputs=outputs)
            record_stage("viz", "skipped", None, None, None, outputs, log_path)
        else:
            (viz_dir / "figures").mkdir(parents=True, exist_ok=True)

            logger.progress("viz", "started")
            start_ts = now_ts()
            start_time = time.time()

            cmd = [
                sys.executable, "make_viz.py",
                "--company_csv", str(score_dir / "company_scores.csv"),
                "--review_csv", str(score_dir / "review_scores.csv"),
                "--per_company_dir", str(score_dir / "per_company"),
                "--out_dir", str(viz_dir / "figures"),
            ]
            log_path = viz_dir / "viz.log"
            rc = run_command(cmd, log_path, repo_root)

            end_ts = now_ts()
            duration = time.time() - start_time
            outputs = list_files_rel(run_dir, viz_dir)

            if rc != 0:
                logger.progress("viz", "failed", returncode=rc, log=str(log_path))
                record_stage("viz", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"viz failed (rc={rc})")

            logger.progress("viz", "completed", outputs=outputs)
            record_stage("viz", "completed", start_ts, end_ts, duration, outputs, log_path)
    except Exception as e:
        if not stage_fail_or_continue("viz", e):
            return 1

    # Optional intermediate cleanup (only after everything that depends on it is done)
    # - scrape is only needed for clean
    # - clean is only needed for extract
    # - extract is only needed for score
    maybe_cleanup_stage(scrape_dir)
    maybe_cleanup_stage(clean_dir)
    maybe_cleanup_stage(extract_dir)

    finalize_report(run_dir, meta_dir, stage_results, errors)
    logger.log("[done] pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
