from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import subprocess
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

from backend.job_store import JobStore
from backend.settings import settings
from csv_safety import sanitize_csv_row

_CACHE_LOG_LOCK = threading.Lock()


def safe_slug(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text).strip())
    s = s.strip("_").lower()
    return s or "job"


def _company_match_key(text: str) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def resolve_company_dir(review_data_dir: Path, company_id: str) -> Path | None:
    wanted = str(company_id or "").strip()
    if not wanted or not review_data_dir.exists():
        return None

    dirs = [p for p in review_data_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None

    for p in dirs:
        if p.name == wanted:
            return p

    wanted_lower = wanted.lower()
    wanted_slug = safe_slug(wanted)
    wanted_key = _company_match_key(wanted)

    for p in dirs:
        if p.name.lower() == wanted_lower:
            return p

    for p in dirs:
        if safe_slug(p.name) == wanted_slug:
            return p

    for p in dirs:
        if _company_match_key(p.name) == wanted_key:
            return p

    return None


def reviews_csv_from_json(json_path: Path, out_csv: Path) -> Path:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in JSON: {json_path}")

    rows: list[dict[str, Any]] = [r for r in raw if isinstance(r, dict)]
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

    first_fields = [f for f in preferred if any(f in r for r in rows)]
    fields = first_fields + discovered
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            rec: dict[str, Any] = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (dict, list)):
                    rec[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    rec[k] = ""
                else:
                    rec[k] = v
            writer.writerow(sanitize_csv_row(rec))

    return out_csv


def resolve_reviews_csv(company_dir: Path, job_dir: Path) -> Path:
    csv_path = company_dir / "reviews.csv"
    if csv_path.exists():
        return csv_path

    json_path = company_dir / "reviews.json"
    if json_path.exists():
        generated = job_dir / "input_reviews_from_json.csv"
        return reviews_csv_from_json(json_path, generated)

    raise FileNotFoundError(
        f"No reviews.csv or reviews.json found in cache folder: {company_dir}"
    )


def make_run_bundle(run_dir: Path) -> Path:
    out_dir = run_dir / "bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "results.zip"

    include_rel = [
        "00_meta",
        "01_scrape",
        "02_clean",
        "03_extract",
        "04_score",
        "05_viz",
        "99_logs",
    ]

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include_rel:
            src = run_dir / rel
            if not src.exists():
                continue
            if src.is_file():
                zf.write(src, arcname=str(src.relative_to(run_dir)))
                continue
            for fp in src.rglob("*"):
                if fp.is_file():
                    zf.write(fp, arcname=str(fp.relative_to(run_dir)))
    return out_zip


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _load_run_report(run_dir: Path) -> dict[str, Any] | None:
    report_path = run_dir / "00_meta" / "run_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_output_summary(run_dir: Path) -> dict[str, Any]:
    report = _load_run_report(run_dir) or {}
    stages = report.get("stages") if isinstance(report, dict) else {}
    counts = report.get("counts") if isinstance(report, dict) else None
    errors = report.get("errors") if isinstance(report, dict) else None

    stage_statuses: dict[str, Any] = {}
    stage_outputs: dict[str, Any] = {}
    if isinstance(stages, dict):
        for stage_name, info in stages.items():
            if not isinstance(info, dict):
                continue
            stage_statuses[str(stage_name)] = info.get("status")
            outs = info.get("outputs")
            if isinstance(outs, list):
                stage_outputs[str(stage_name)] = outs
            else:
                stage_outputs[str(stage_name)] = []

    key_files = {
        "run_report_json": run_dir / "00_meta" / "run_report.json",
        "review_scores_csv": run_dir / "04_score" / "review_scores.csv",
        "company_scores_csv": run_dir / "04_score" / "company_scores.csv",
        "figures_dir": run_dir / "05_viz" / "figures",
    }
    key_paths = {k: str(v) for k, v in key_files.items()}
    key_exists = {k: v.exists() for k, v in key_files.items()}

    return {
        "counts": counts,
        "errors": errors,
        "stage_statuses": stage_statuses,
        "stage_outputs": stage_outputs,
        "key_paths": key_paths,
        "key_exists": key_exists,
    }


def _missing_required_viz(run_dir: Path) -> list[str]:
    required = [
        run_dir / "05_viz" / "figures" / "04_fulfillment_vs_hindrance_stacked.png",
        run_dir / "05_viz" / "figures" / "06_company_radar_all.png",
    ]
    return [str(p.relative_to(run_dir)).replace("\\", "/") for p in required if not p.exists()]


def _append_cache_usage_log(entry: dict[str, Any]) -> None:
    log_path = settings.CACHE_USAGE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _CACHE_LOG_LOCK:
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")


def run_cache_job(store: JobStore, job_id: str, company_id: str) -> None:
    started_at = time.time()
    started_iso = _now_iso()
    resolved_company_name: str | None = None
    company_dir: Path | None = None
    input_csv: Path | None = None
    run_dir: Path | None = None
    bundle_path: Path | None = None

    def write_audit(status: str, rc: int | None, message: str) -> None:
        ended_at = time.time()
        ended_iso = _now_iso()
        payload: dict[str, Any] = {
            "ts": ended_iso,
            "job_id": job_id,
            "company_id_requested": company_id,
            "company_id_resolved": resolved_company_name,
            "cache_company_dir": str(company_dir) if company_dir else None,
            "cache_input_csv": str(input_csv) if input_csv else None,
            "status": status,
            "rc": rc,
            "message": message,
            "started_at": started_iso,
            "ended_at": ended_iso,
            "duration_sec": round(ended_at - started_at, 3),
            "run_dir": str(run_dir) if run_dir else None,
            "bundle_path": str(bundle_path) if bundle_path else None,
        }
        if run_dir and run_dir.exists():
            payload["output_summary"] = _collect_output_summary(run_dir)
        _append_cache_usage_log(payload)

    try:
        review_root = settings.REVIEW_DATA_DIR
        company_dir = resolve_company_dir(review_root, company_id)
        if company_dir is None:
            msg = f"Company ID not found under '{review_root}': {company_id}"
            store.update_status(
                job_id,
                status="failed",
                message=msg,
                rc=1,
            )
            write_audit(status="failed", rc=1, message=msg)
            return

        resolved_company_name = company_dir.name
        job_slug = safe_slug(company_dir.name)
        run_dir = settings.RUNS_DIR / job_slug / job_id
        store.update_status(
            job_id,
            status="running",
            message="Preparing cached input",
            company_id_resolved=company_dir.name,
            job_slug=job_slug,
            run_dir=str(run_dir),
        )

        input_csv = resolve_reviews_csv(company_dir, store.job_dir(job_id))

        cmd = [
            str(settings.PYTHON_EXE),
            "pipeline.py",
            "--job",
            company_dir.name,
            "--run-root",
            str(settings.RUNS_DIR),
            "--run-id",
            job_id,
            "--skip-scrape",
            "--raw-csv",
            str(input_csv),
            "--goal-dict",
            str(settings.GOAL_DICT),
        ]

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        log_path = store.log_path(job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write("[runner] command:\n  " + " ".join(cmd) + "\n\n")
            lf.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=str(settings.REPO_ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            store.update_status(
                job_id,
                status="running",
                message="Pipeline running",
                pid=proc.pid,
                pipeline_cmd=cmd,
            )

            rc = proc.wait()

        if rc != 0:
            msg = f"Pipeline failed (rc={rc})"
            store.update_status(
                job_id,
                status="failed",
                message=msg,
                rc=rc,
                pid=None,
            )
            write_audit(status="failed", rc=rc, message=msg)
            return

        if not run_dir.exists():
            msg = f"Pipeline succeeded but run_dir not found: {run_dir}"
            store.update_status(
                job_id,
                status="failed",
                message=msg,
                rc=2,
                pid=None,
            )
            write_audit(status="failed", rc=2, message=msg)
            return

        missing_viz = _missing_required_viz(run_dir)
        if missing_viz:
            msg = "Pipeline completed but required visualizations are missing: " + ", ".join(missing_viz)
            store.update_status(
                job_id,
                status="failed",
                message=msg,
                rc=3,
                pid=None,
            )
            write_audit(status="failed", rc=3, message=msg)
            return

        bundle_path = make_run_bundle(run_dir)
        store.update_status(
            job_id,
            status="succeeded",
            message="Pipeline completed",
            rc=0,
            pid=None,
            bundle_path=str(bundle_path),
        )
        store.append_log(job_id, f"[runner] bundle: {bundle_path}")
        write_audit(status="succeeded", rc=0, message="Pipeline completed")

    except Exception as exc:  # noqa: BLE001
        msg = f"Runner exception: {exc}"
        store.append_log(job_id, "[runner][error] " + repr(exc))
        store.append_log(job_id, traceback.format_exc())
        store.update_status(
            job_id,
            status="failed",
            message=msg,
            rc=99,
            pid=None,
        )
        write_audit(status="failed", rc=99, message=msg)
