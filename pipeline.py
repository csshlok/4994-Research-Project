from __future__ import annotations

import argparse
import ctypes
import glob
import json
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


def canonicalize_reviews_url(url: str) -> str:
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
    qd["filter.iso3Language"] = ["eng"]
    qd["sort.sortType"] = ["RD"]
    qd["sort.ascending"] = ["false"]

    query = urlencode(qd, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, path_final, "", query, ""))


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
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command:\n  " + " ".join(cmd) + "\n\n")
        res = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    return res.returncode


def wait_for_log_marker(log_path: Path, marker: str, proc: subprocess.Popen,
                        timeout_sec: float = 60.0) -> bool:
    start = time.time()
    while (time.time() - start) < timeout_sec:
        if proc.poll() is not None:
            return False
        try:
            if log_path.exists():
                data = log_path.read_text(encoding="utf-8", errors="ignore")
                if marker in data:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


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
                if ok:
                    reason = {
                        "first_page": "first page load",
                        "challenge": "challenge detected",
                        "base_resolved": "base resolved",
                    }.get(trigger, "pause trigger")
                    if pause_until_enter:
                        logger.log(f"[scrape] Paused after {reason}; waiting for Enter.")
                        pause_for_user("Solve any captcha/login, then press Enter to continue...")
                    if pause_seconds and pause_seconds > 0:
                        logger.log(f"[scrape] Pausing {pause_seconds:.1f}s after {reason}.")
                        time.sleep(float(pause_seconds))
                    ok, err = resume_process(proc.pid)
                    if not ok:
                        logger.log(f"[scrape] Resume failed; scraper may still be paused. {err}")
                else:
                    logger.log(f"[scrape] Suspend failed; skipping pause. {err}")
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
        res = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False
        )
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

    ap.add_argument("--raw-csv", default=None, help="Raw CSV path or glob (if skipping scrape).")
    ap.add_argument("--clean-glob", default=None, help="Cleaned CSV glob (if skipping clean).")
    ap.add_argument("--features-dir", default=None, help="Features dir (if skipping extract).")
    ap.add_argument("--scored-out-dir", default=None, help="Scoring output dir (if skipping score).")
    ap.add_argument("--goal-dict", default=None, help="Goal dictionary path (optional).")

    ap.add_argument("--pages", type=int, default=3, help="Number of review pages to collect.")
    ap.add_argument("--page-delay", type=float, default=3.0, help="Seconds to wait between pages.")
    ap.add_argument("--headless", action="store_true", help="Run Chrome headless.")
    ap.add_argument("--pause-seconds", type=float, default=0.0,
                    help="Seconds to wait after the first page loads (or on challenge).")
    ap.add_argument("--pause-until-enter", action="store_true",
                    help="Pause after the first page loads (or on challenge) until Enter is pressed.")
    ap.add_argument("--chrome-binary", default=None, help="Path to Chrome binary.")
    ap.add_argument("--profile-dir", default=None, help="Chrome profile dir (default: chrome-profile).")
    ap.add_argument("--timeout", type=int, default=1600, help="Overall scraper timeout (seconds).")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_scrape and not args.url:
        raise ValueError("--url is required unless --skip-scrape is set.")
    if args.skip_scrape and not args.raw_csv:
        raise ValueError("--raw-csv is required when --skip-scrape is set.")
    if args.skip_clean and not args.clean_glob:
        raise ValueError("--clean-glob is required when --skip-clean is set.")
    if args.skip_extract and not args.features_dir:
        raise ValueError("--features-dir is required when --skip-extract is set.")
    if args.skip_score and not args.scored_out_dir:
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
    resolved_url = canonicalize_reviews_url(args.url) if args.url else None

    args_path = meta_dir / "pipeline_args.json"
    args_payload = vars(args).copy()
    args_payload["job_slug"] = job_slug
    args_payload["run_id"] = run_id
    args_payload["run_dir"] = str(run_dir)
    args_payload["profile_dir_resolved"] = resolved_profile_dir
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
            log_path.write_text(
                "scrape skipped\ninputs:\n" + "\n".join([str(p) for p in raw_files]) + "\n",
                encoding="utf-8"
            )
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
                "--page-delay", str(args.page_delay),
                "--timeout", str(args.timeout),
                "--out", str(reviews_json),
                "--csv", str(reviews_csv),
            ]
            if args.headless:
                cmd.append("--headless")
            if args.chrome_binary:
                cmd += ["--chrome-binary", args.chrome_binary]
            cmd += ["--profile-dir", profile_dir]

            log_path = scrape_dir / "scrape.log"
            rc = run_scraper_command(
                cmd,
                log_path,
                repo_root,
                args.pause_seconds,
                args.pause_until_enter,
                logger
            )

            end_ts = now_ts()
            duration = time.time() - start_time
            outputs = list_files_rel(run_dir, scrape_dir)

            if rc != 0:
                logger.progress("scrape", "failed", returncode=rc, log=str(log_path))
                record_stage("scrape", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"scrape failed (rc={rc})")

            logger.progress("scrape", "completed", outputs=outputs)
            record_stage("scrape", "completed", start_ts, end_ts, duration, outputs, log_path)
    except Exception as e:
        errors.append(f"scrape: {e}")
        logger.log(f"[scrape][error] {e}")
        if not args.continue_on_error:
            finalize_report(run_dir, meta_dir, stage_results, errors)
            return 1

    # ---------------- Clean ----------------
    try:
        if args.skip_clean:
            clean_files = resolve_files(args.clean_glob, dir_pattern="*.csv")
            if not clean_files:
                raise FileNotFoundError(f"No files matched: {args.clean_glob}")
            copy_files(clean_files, clean_dir)
            outputs = list_files_rel(run_dir, clean_dir)
            log_path = clean_dir / "clean.log"
            log_path.write_text(
                "clean skipped\ninputs:\n" + "\n".join([str(p) for p in clean_files]) + "\n",
                encoding="utf-8"
            )
            report_path = clean_dir / "clean_report.json"
            if not report_path.exists():
                report = {
                    "raw_dir": None,
                    "pattern": args.clean_glob,
                    "job_slug": None,
                    "files": [str(p) for p in clean_files],
                    "totals": None,
                    "note": "clean skipped; inputs staged by pipeline",
                }
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.progress("clean", "skipped", reason="user_skip", outputs=outputs)
            record_stage("clean", "skipped", None, None, None, outputs, log_path,
                         extra={"input_sources": [str(p) for p in clean_files]})
        else:
            logger.progress("clean", "started")
            start_ts = now_ts()
            start_time = time.time()

            raw_csvs = sorted(scrape_dir.glob("*.csv"))
            clean_job = job_slug if len(raw_csvs) == 1 else None
            if len(raw_csvs) == 1 and (scrape_dir / "reviews.csv").exists():
                pattern = str(scrape_dir / "reviews.csv")
            else:
                pattern = str(scrape_dir / "*.csv")

            cmd = [
                sys.executable, "data_cleaner.py",
                "--raw-dir", str(scrape_dir),
                "--out-dir", str(clean_dir),
                "--pattern", pattern,
                "--report-json", str(clean_dir / "clean_report.json"),
            ]
            if clean_job:
                cmd += ["--job", clean_job]

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
        errors.append(f"clean: {e}")
        logger.log(f"[clean][error] {e}")
        if not args.continue_on_error:
            finalize_report(run_dir, meta_dir, stage_results, errors)
            return 1

    # ---------------- Extract ----------------
    try:
        if args.skip_extract:
            copy_dir(Path(args.features_dir), extract_dir)
            outputs = list_files_rel(run_dir, extract_dir)
            log_path = extract_dir / "extract.log"
            log_path.write_text(f"extract skipped\ninput_dir: {args.features_dir}\n", encoding="utf-8")
            logger.progress("extract", "skipped", reason="user_skip", outputs=outputs)
            record_stage("extract", "skipped", None, None, None, outputs, log_path,
                         extra={"input_source_dir": args.features_dir})
        else:
            logger.progress("extract", "started")
            start_ts = now_ts()
            start_time = time.time()

            clean_pattern = str(clean_dir / "reviews_*.csv")
            if not glob.glob(clean_pattern):
                clean_pattern = str(clean_dir / "*.csv")
            cmd = [
                sys.executable, "extraction.py",
                "--in", clean_pattern,
                "--out", str(extract_dir),
            ]

            log_path = extract_dir / "extract.log"
            rc = run_command(cmd, log_path, repo_root)

            end_ts = now_ts()
            duration = time.time() - start_time
            if rc == 0:
                tfidf_src = extract_dir / "tfidf_reviews.npz"
                tfidf_dst = extract_dir / "tfidf.npz"
                if tfidf_src.exists() and not tfidf_dst.exists():
                    shutil.copy2(tfidf_src, tfidf_dst)
                vocab_src = extract_dir / "tfidf_vocab.json"
                vocab_dst = extract_dir / "vocab.json"
                if vocab_src.exists() and not vocab_dst.exists():
                    shutil.copy2(vocab_src, vocab_dst)

            outputs = list_files_rel(run_dir, extract_dir)

            if rc != 0:
                logger.progress("extract", "failed", returncode=rc, log=str(log_path))
                record_stage("extract", "failed", start_ts, end_ts, duration, outputs, log_path)
                raise RuntimeError(f"extract failed (rc={rc})")

            logger.progress("extract", "completed", outputs=outputs)
            record_stage("extract", "completed", start_ts, end_ts, duration, outputs, log_path)
    except Exception as e:
        errors.append(f"extract: {e}")
        logger.log(f"[extract][error] {e}")
        if not args.continue_on_error:
            finalize_report(run_dir, meta_dir, stage_results, errors)
            return 1

    # ---------------- Score ----------------
    try:
        if args.skip_score:
            copy_dir(Path(args.scored_out_dir), score_dir)
            outputs = list_files_rel(run_dir, score_dir)
            log_path = score_dir / "score.log"
            log_path.write_text(f"score skipped\ninput_dir: {args.scored_out_dir}\n", encoding="utf-8")
            logger.progress("score", "skipped", reason="user_skip", outputs=outputs)
            record_stage("score", "skipped", None, None, None, outputs, log_path,
                         extra={"input_source_dir": args.scored_out_dir})
        else:
            logger.progress("score", "started")
            start_ts = now_ts()
            start_time = time.time()

            goal_dict = args.goal_dict or str((repo_root / "config" / "goal_dict.json"))
            cmd = [
                sys.executable, "scorer.py",
                "--features-dir", str(extract_dir),
                "--goal-dict", goal_dict,
                "--out-dir", str(score_dir),
            ]

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
        errors.append(f"score: {e}")
        logger.log(f"[score][error] {e}")
        if not args.continue_on_error:
            finalize_report(run_dir, meta_dir, stage_results, errors)
            return 1

    # ---------------- Viz ----------------
    try:
        if args.skip_viz:
            log_path = viz_dir / "viz.log"
            log_path.write_text("viz skipped\n", encoding="utf-8")
            logger.progress("viz", "skipped", reason="user_skip", outputs=[])
            record_stage("viz", "skipped", None, None, None, [], log_path)
        else:
            logger.progress("viz", "started")
            start_ts = now_ts()
            start_time = time.time()

            fig_dir = viz_dir / "figures"
            cmd = [
                sys.executable, "make_viz.py",
                "--company_csv", str(score_dir / "company_scores.csv"),
                "--review_csv", str(score_dir / "review_scores.csv"),
                "--per_company_dir", str(score_dir / "per_company"),
                "--out_dir", str(fig_dir),
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
        errors.append(f"viz: {e}")
        logger.log(f"[viz][error] {e}")
        if not args.continue_on_error:
            finalize_report(run_dir, meta_dir, stage_results, errors)
            return 1

    finalize_report(run_dir, meta_dir, stage_results, errors)
    logger.log("[done] pipeline finished")

    if not args.keep_intermediate:
        for d in [scrape_dir, clean_dir, extract_dir]:
            if d.exists():
                shutil.rmtree(d)
                logger.log(f"[cleanup] removed {d}")
    return 0


def finalize_report(run_dir: Path, meta_dir: Path,
                    stage_results: Dict[str, Dict[str, Any]],
                    errors: List[str]) -> None:
    report = {
        "run_dir": str(run_dir),
        "stages": stage_results,
        "errors": errors,
    }
    args_path = meta_dir / "pipeline_args.json"
    if args_path.exists():
        try:
            report["pipeline_args"] = json.loads(args_path.read_text(encoding="utf-8"))
        except Exception:
            report["pipeline_args"] = None

    # Counts and schema
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


if __name__ == "__main__":
    sys.exit(main())
