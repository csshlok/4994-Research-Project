from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from csv_safety import sanitize_csv_row


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW_ROOT = ROOT / "review data"
DEFAULT_SCORE_ROOT = ROOT / "company scores"
DEFAULT_RUN_ROOT = ROOT / "runs" / "company_score_precompute"
DEFAULT_GOAL_DICT = ROOT / "config" / "goal_dict.json"
DEFAULT_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_EXCLUDED_COMPANIES = {"wells_fargo", "wells fargo"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_slug(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text).strip())
    s = s.strip("_").lower()
    return s or "company"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reviews_csv_from_json(json_path: Path, out_csv: Path) -> Path:
    raw = load_json(json_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in JSON: {json_path}")

    rows: list[dict[str, Any]] = [row for row in raw if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No review objects found in JSON: {json_path}")

    preferred = [
        "review_id",
        "title",
        "body",
        "pros",
        "cons",
        "rating",
        "date",
        "role",
        "location",
        "employmentStatus",
        "_source",
    ]
    discovered: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in preferred and key not in discovered:
                discovered.append(key)

    fields = [f for f in preferred if any(f in row for row in rows)] + discovered
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            rec: dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    rec[key] = json.dumps(value, ensure_ascii=False)
                elif value is None:
                    rec[key] = ""
                else:
                    rec[key] = value
            writer.writerow(sanitize_csv_row(rec))
    return out_csv


def resolve_input_csv(company_dir: Path, work_dir: Path) -> Path:
    csv_path = company_dir / "reviews.csv"
    if csv_path.exists():
        return csv_path

    json_path = company_dir / "reviews.json"
    if json_path.exists():
        return reviews_csv_from_json(json_path, work_dir / "input_reviews_from_json.csv")

    raise FileNotFoundError(f"No reviews.csv or reviews.json found in {company_dir}")


def discover_companies(review_root: Path, requested: list[str] | None) -> list[Path]:
    if not review_root.exists():
        raise FileNotFoundError(f"Missing review root: {review_root}")

    dirs = sorted(
        [
            p
            for p in review_root.iterdir()
            if p.is_dir() and p.name.casefold() not in DEFAULT_EXCLUDED_COMPANIES
        ],
        key=lambda p: p.name.lower(),
    )
    if not requested:
        return dirs

    wanted = {name.casefold(): name for name in requested}
    by_exact = {p.name.casefold(): p for p in dirs}
    missing = [original for key, original in wanted.items() if key not in by_exact]
    if missing:
        raise FileNotFoundError(f"Company folder(s) not found under {review_root}: {', '.join(missing)}")
    return [by_exact[key] for key in wanted.keys()]


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return str(dst)


def collect_outputs(run_dir: Path, score_dir: Path) -> dict[str, Any]:
    copied: dict[str, Any] = {}

    copied["review_scores_csv"] = copy_if_exists(
        run_dir / "04_score" / "review_scores.csv",
        score_dir / "review_scores.csv",
    )
    copied["company_scores_csv"] = copy_if_exists(
        run_dir / "04_score" / "company_scores.csv",
        score_dir / "company_scores.csv",
    )
    copied["score_run_report_json"] = copy_if_exists(
        run_dir / "04_score" / "run_report.json",
        score_dir / "score_run_report.json",
    )
    copied["per_company_dir"] = copy_if_exists(
        run_dir / "04_score" / "per_company",
        score_dir / "per_company",
    )
    copied["pipeline_run_report_json"] = copy_if_exists(
        run_dir / "00_meta" / "run_report.json",
        score_dir / "pipeline_run_report.json",
    )
    copied["pipeline_args_json"] = copy_if_exists(
        run_dir / "00_meta" / "pipeline_args.json",
        score_dir / "pipeline_args.json",
    )

    clean_files = sorted((run_dir / "02_clean").glob("*.csv"))
    if clean_files:
        copied["cleaned_reviews_csv"] = copy_if_exists(clean_files[0], score_dir / "cleaned_reviews.csv")
    copied["clean_report_json"] = copy_if_exists(
        run_dir / "02_clean" / "clean_report.json",
        score_dir / "clean_report.json",
    )
    copied["extract_config_json"] = copy_if_exists(
        run_dir / "03_extract" / "config.json",
        score_dir / "extract_config.json",
    )

    return copied


def has_required_outputs(score_dir: Path) -> bool:
    return (score_dir / "review_scores.csv").exists() and (score_dir / "company_scores.csv").exists()


def build_pipeline_cmd(
    python_exe: str,
    company_name: str,
    input_csv: Path,
    run_root: Path,
    run_id: str,
    goal_dict: Path,
    with_viz: bool,
) -> list[str]:
    cmd = [
        python_exe,
        "pipeline.py",
        "--job",
        company_name,
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--skip-scrape",
        "--raw-csv",
        str(input_csv),
        "--goal-dict",
        str(goal_dict),
        "--keep-intermediate",
        "true",
        "--compress-artifacts",
        "false",
    ]
    if not with_viz:
        cmd.append("--skip-viz")
    return cmd


def precompute_company(
    company_dir: Path,
    score_root: Path,
    run_root: Path,
    goal_dict: Path,
    python_exe: str,
    force: bool,
    with_viz: bool,
    keep_runs: bool,
    dry_run: bool,
) -> dict[str, Any]:
    company_name = company_dir.name
    slug = safe_slug(company_name)
    score_dir = score_root / company_name
    score_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    status_path = score_dir / "precompute_status.json"
    payload: dict[str, Any] = {
        "company": company_name,
        "company_dir": str(company_dir),
        "score_dir": str(score_dir),
        "started_at": now_iso(),
        "status": "running",
    }
    write_json(status_path, payload)

    if not force and has_required_outputs(score_dir):
        payload.update(
            {
                "status": "skipped",
                "message": "Required score outputs already exist.",
                "ended_at": now_iso(),
                "duration_sec": round(time.time() - started, 3),
            }
        )
        write_json(status_path, payload)
        return payload

    work_dir = score_dir / "_work"
    run_id = slug
    run_dir = run_root / run_id

    try:
        input_csv = resolve_input_csv(company_dir, work_dir)
        cmd = build_pipeline_cmd(
            python_exe=python_exe,
            company_name=company_name,
            input_csv=input_csv,
            run_root=run_root,
            run_id=run_id,
            goal_dict=goal_dict,
            with_viz=with_viz,
        )
        payload.update(
            {
                "input_csv": str(input_csv),
                "run_dir": str(run_dir),
                "command": cmd,
            }
        )
        write_json(status_path, payload)

        if dry_run:
            payload.update(
                {
                    "status": "dry_run",
                    "message": "Command was prepared but not executed.",
                    "ended_at": now_iso(),
                    "duration_sec": round(time.time() - started, 3),
                }
            )
            write_json(status_path, payload)
            return payload

        run_root.mkdir(parents=True, exist_ok=True)
        if run_dir.exists():
            shutil.rmtree(run_dir)

        log_path = score_dir / "precompute.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("Command:\n  " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        payload["returncode"] = proc.returncode
        payload["log_path"] = str(log_path)
        if proc.returncode != 0:
            payload.update(
                {
                    "status": "failed",
                    "message": f"pipeline.py failed with return code {proc.returncode}",
                    "ended_at": now_iso(),
                    "duration_sec": round(time.time() - started, 3),
                }
            )
            write_json(status_path, payload)
            return payload

        copied = collect_outputs(run_dir, score_dir)
        missing = [
            name
            for name in ["review_scores_csv", "company_scores_csv"]
            if not copied.get(name)
        ]
        if missing:
            payload.update(
                {
                    "status": "failed",
                    "message": "Pipeline completed but required copied outputs are missing: "
                    + ", ".join(missing),
                    "copied_outputs": copied,
                    "ended_at": now_iso(),
                    "duration_sec": round(time.time() - started, 3),
                }
            )
            write_json(status_path, payload)
            return payload

        topic_cmd = [
            python_exe,
            "generate_topic_artifacts.py",
            "--score-root",
            str(score_root),
            "--goal-dict",
            str(goal_dict),
            "--company",
            company_name,
        ]
        topic_log_path = score_dir / "topic_precompute.log"
        with topic_log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("Command:\n  " + " ".join(topic_cmd) + "\n\n")
            log.flush()
            topic_proc = subprocess.run(
                topic_cmd,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        copied["topic_summary_csv"] = str(score_dir / "topic_summary.csv") if (score_dir / "topic_summary.csv").exists() else None
        copied["topic_assignments_csv"] = str(score_dir / "topic_assignments.csv") if (score_dir / "topic_assignments.csv").exists() else None
        copied["topic_precompute_log"] = str(topic_log_path)
        payload["topic_returncode"] = topic_proc.returncode
        if topic_proc.returncode != 0:
            payload.update(
                {
                    "status": "failed",
                    "message": f"topic generation failed with return code {topic_proc.returncode}",
                    "copied_outputs": copied,
                    "ended_at": now_iso(),
                    "duration_sec": round(time.time() - started, 3),
                }
            )
            write_json(status_path, payload)
            return payload

        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        if not keep_runs and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        payload.update(
            {
                "status": "succeeded",
                "message": "Company scores precomputed.",
                "copied_outputs": copied,
                "ended_at": now_iso(),
                "duration_sec": round(time.time() - started, 3),
            }
        )
        write_json(status_path, payload)
        return payload

    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "message": str(exc),
                "ended_at": now_iso(),
                "duration_sec": round(time.time() - started, 3),
            }
        )
        write_json(status_path, payload)
        return payload


def update_root_manifest(score_root: Path, review_root: Path, results: list[dict[str, Any]]) -> None:
    companies = sorted(
        [
            p.name
            for p in review_root.iterdir()
            if p.is_dir() and p.name.casefold() not in DEFAULT_EXCLUDED_COMPANIES
        ],
        key=str.lower,
    )
    summary = {
        "source_root": str(review_root),
        "score_root": str(score_root),
        "updated_at": now_iso(),
        "companies": companies,
        "latest_run": {
            "total": len(results),
            "succeeded": sum(1 for r in results if r.get("status") == "succeeded"),
            "skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "dry_run": sum(1 for r in results if r.get("status") == "dry_run"),
            "results": [
                {
                    "company": r.get("company"),
                    "status": r.get("status"),
                    "message": r.get("message"),
                    "duration_sec": r.get("duration_sec"),
                }
                for r in results
            ],
        },
    }
    write_json(score_root / "manifest.json", summary)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Precompute per-company score caches from review data folders."
    )
    ap.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    ap.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--goal-dict", default=str(DEFAULT_GOAL_DICT))
    ap.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python executable used to run pipeline.py.",
    )
    ap.add_argument(
        "--company",
        action="append",
        default=None,
        help="Company folder to precompute. Repeat to process multiple. Defaults to all companies.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N selected companies.")
    ap.add_argument("--force", action="store_true", help="Recompute even if required outputs already exist.")
    ap.add_argument("--with-viz", action="store_true", help="Also run visualization stage.")
    ap.add_argument("--keep-runs", action="store_true", help="Keep temporary pipeline run folders.")
    ap.add_argument("--dry-run", action="store_true", help="Prepare commands and status files without running.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    review_root = Path(args.review_root).resolve()
    score_root = Path(args.score_root).resolve()
    run_root = Path(args.run_root).resolve()
    goal_dict = Path(args.goal_dict).resolve()

    if not goal_dict.exists():
        raise FileNotFoundError(f"Missing goal dictionary: {goal_dict}")

    companies = discover_companies(review_root, args.company)
    if args.limit is not None:
        companies = companies[: max(0, int(args.limit))]

    score_root.mkdir(parents=True, exist_ok=True)
    for company in companies:
        (score_root / company.name).mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, company_dir in enumerate(companies, start=1):
        print(f"[{idx}/{len(companies)}] {company_dir.name}", flush=True)
        result = precompute_company(
            company_dir=company_dir,
            score_root=score_root,
            run_root=run_root,
            goal_dict=goal_dict,
            python_exe=str(args.python),
            force=bool(args.force),
            with_viz=bool(args.with_viz),
            keep_runs=bool(args.keep_runs),
            dry_run=bool(args.dry_run),
        )
        results.append(result)
        print(f"  -> {result.get('status')}: {result.get('message', '')}", flush=True)

    update_root_manifest(score_root, review_root, results)
    failed = [r for r in results if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
