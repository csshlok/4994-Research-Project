from __future__ import annotations

import gzip
import json
import mimetypes
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Allow launching this file directly (for example, via an IDE "Run file" action).
if __package__ in {None, ""}:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from backend.job_store import JobStore
from backend.pipeline_runner import run_cache_job
from backend.settings import settings


class RunRequest(BaseModel):
    mode: str = "cache"
    company_id: str | None = None
    company_name: str | None = None


def _spawn(target, *args) -> None:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()


def _gz_path(path: Path) -> Path:
    return path.with_name(path.name + ".gz")


def _read_text_maybe_gz(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    gz = _gz_path(path)
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def _read_json_maybe_gz(path: Path) -> dict[str, Any] | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    gz = _gz_path(path)
    if gz.exists():
        try:
            with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _iter_gzip_bytes(path: Path, chunk_size: int = 1024 * 1024):
    with gzip.open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


ACTIVE_JOB_STATUSES = {"queued", "running"}


def _load_status_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _iter_job_statuses() -> list[tuple[str, dict[str, Any]]]:
    jobs_root = settings.JOBS_DIR
    if not jobs_root.exists():
        return []

    out: list[tuple[str, dict[str, Any]]] = []
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        status = _load_status_file(job_dir / "status.json")
        if status:
            out.append((job_id, status))
    return out


def _count_jobs_by_status() -> dict[str, int]:
    counts = {"queued": 0, "running": 0}
    for _job_id, status in _iter_job_statuses():
        state = str(status.get("status") or "").strip().lower()
        if state in counts:
            counts[state] += 1
    return counts


def _pick_next_queued_job() -> tuple[str, dict[str, Any]] | None:
    queued: list[tuple[str, dict[str, Any]]] = []
    for job_id, status in _iter_job_statuses():
        state = str(status.get("status") or "").strip().lower()
        if state == "queued":
            queued.append((job_id, status))
    if not queued:
        return None

    queued.sort(key=lambda item: float(item[1].get("created_at") or 0.0))
    return queued[0]


def _job_dispatch_loop() -> None:
    sleep_s = float(settings.JOB_DISPATCH_INTERVAL_SECONDS)
    while True:
        try:
            counts = _count_jobs_by_status()
            if counts["running"] > 0:
                time.sleep(sleep_s)
                continue

            nxt = _pick_next_queued_job()
            if nxt is None:
                time.sleep(sleep_s)
                continue

            job_id, status = nxt
            company = str(status.get("company_id_requested") or "").strip()
            if not company:
                store.update_status(
                    job_id,
                    status="failed",
                    message="Missing company_id_requested in queued job.",
                    rc=98,
                )
                time.sleep(0.1)
                continue

            # Runs synchronously here; this enforces single-job concurrency.
            run_cache_job(store, job_id, company)
            continue
        except Exception:
            pass

        time.sleep(sleep_s)


def _cleanup_jobs_and_collect_active_runs(now_ts: float) -> set[Path]:
    active_runs: set[Path] = set()
    jobs_root = settings.JOBS_DIR
    if not jobs_root.exists():
        return active_runs

    ttl_seconds = float(settings.JOB_RETENTION_HOURS) * 3600.0
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue

        status_path = job_dir / "status.json"
        status = _load_status_file(status_path)
        state = str(status.get("status") or "").strip().lower()
        run_dir_val = status.get("run_dir")
        if state in ACTIVE_JOB_STATUSES and isinstance(run_dir_val, str) and run_dir_val.strip():
            active_runs.add(Path(run_dir_val).resolve())

        updated_at = status.get("updated_at")
        created_at = status.get("created_at")
        if isinstance(updated_at, (int, float)):
            age_ref = float(updated_at)
        elif isinstance(created_at, (int, float)):
            age_ref = float(created_at)
        else:
            age_ref = float(job_dir.stat().st_mtime)

        if state in ACTIVE_JOB_STATUSES:
            continue
        if (now_ts - age_ref) >= ttl_seconds:
            shutil.rmtree(job_dir, ignore_errors=True)

    return active_runs


def _cleanup_runs(now_ts: float, active_runs: set[Path]) -> None:
    runs_root = settings.RUNS_DIR
    if not runs_root.exists():
        return

    ttl_seconds = float(settings.RUN_RETENTION_HOURS) * 3600.0
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        run_resolved = run_dir.resolve()
        if run_resolved in active_runs:
            continue
        # Prefer job created_at when available so TTL is based on job creation time.
        status_path = settings.JOBS_DIR / run_dir.name / "status.json"
        status = _load_status_file(status_path)
        created_at = status.get("created_at")
        if isinstance(created_at, (int, float)):
            age_ref = float(created_at)
        else:
            age_ref = float(run_dir.stat().st_mtime)
        if (now_ts - age_ref) >= ttl_seconds:
            shutil.rmtree(run_dir, ignore_errors=True)


def _run_ttl_cleanup_once() -> None:
    now_ts = time.time()
    active_runs = _cleanup_jobs_and_collect_active_runs(now_ts)
    _cleanup_runs(now_ts, active_runs)


def _cleanup_loop() -> None:
    while True:
        try:
            _run_ttl_cleanup_once()
        except Exception:
            pass
        time.sleep(float(settings.CLEANUP_INTERVAL_SECONDS))


def _read_progress_snapshot(progress_path: Path) -> dict[str, Any]:
    body = _read_text_maybe_gz(progress_path)
    if not body:
        return {"current_stage": None, "current_status": None, "history": []}

    history: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            history.append(rec)

    current_stage = history[-1].get("stage") if history else None
    current_status = history[-1].get("status") if history else None
    return {
        "current_stage": current_stage,
        "current_status": current_status,
        "history": history,
    }


def _tail_file(path: Path, n: int) -> str:
    body = _read_text_maybe_gz(path)
    if not body:
        return ""
    lines = body.splitlines()
    return "\n".join(lines[-max(1, n):])


app = FastAPI(title="Pipeline Backend", version="1.0.0")
store = JobStore(settings.JOBS_DIR)
_spawn(_cleanup_loop)
_spawn(_job_dispatch_loop)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    counts = _count_jobs_by_status()
    return {
        "ok": True,
        "repo_root": str(settings.REPO_ROOT),
        "review_data_dir": str(settings.REVIEW_DATA_DIR),
        "runs_dir": str(settings.RUNS_DIR),
        "cache_usage_log": str(settings.CACHE_USAGE_LOG),
        "python_exe": str(settings.PYTHON_EXE),
        "run_retention_hours": settings.RUN_RETENTION_HOURS,
        "job_retention_hours": settings.JOB_RETENTION_HOURS,
        "cleanup_interval_seconds": settings.CLEANUP_INTERVAL_SECONDS,
        "job_dispatch_interval_seconds": settings.JOB_DISPATCH_INTERVAL_SECONDS,
        "queue_depth": counts["queued"],
        "running_jobs": counts["running"],
    }


@app.get("/api/companies")
def list_companies() -> dict[str, Any]:
    root = settings.REVIEW_DATA_DIR
    if not root.exists():
        return {"companies": []}

    companies: list[dict[str, Any]] = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
        companies.append(
            {
                "id": p.name,
                "has_csv": (p / "reviews.csv").exists(),
                "has_json": (p / "reviews.json").exists(),
            }
        )
    return {"companies": companies}


@app.post("/api/run")
def run_job(payload: RunRequest) -> dict[str, str]:
    if payload.mode != "cache":
        raise HTTPException(status_code=400, detail="Only mode='cache' is supported right now.")

    company = (payload.company_id or payload.company_name or "").strip()
    if not company:
        raise HTTPException(status_code=400, detail="Provide company_id (or company_name).")

    job_id = uuid.uuid4().hex[:12]
    store.create_job(
        job_id,
        {
            "status": "queued",
            "mode": "cache",
            "company_id_requested": company,
            "message": "Queued",
            "rc": None,
            "pid": None,
            "run_dir": None,
            "bundle_path": None,
        },
    )

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_str = st.get("run_dir")
    if run_dir_str:
        run_dir = Path(run_dir_str)
        progress = _read_progress_snapshot(run_dir / "99_logs" / "progress.jsonl")
        st["stage"] = {
            "current": progress["current_stage"],
            "status": progress["current_status"],
            "history": progress["history"],
        }

        report_path = run_dir / "00_meta" / "run_report.json"
        report = _read_json_maybe_gz(report_path)
        if report is not None:
            st["pipeline_report"] = {
                "counts": report.get("counts"),
                "errors": report.get("errors"),
                "stages": report.get("stages"),
            }
        else:
            st["pipeline_report"] = None
    return st


@app.get("/api/job/{job_id}/log")
def job_log(job_id: str, n: int = Query(default=200, ge=1, le=2000)) -> PlainTextResponse:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    sections: list[str] = []
    runner_tail = store.tail_log(job_id, n=n)
    if runner_tail:
        sections.append("=== runner.log ===")
        sections.append(runner_tail)

    run_dir_str = st.get("run_dir")
    if run_dir_str:
        pipeline_log = Path(run_dir_str) / "99_logs" / "pipeline.log"
        p_tail = _tail_file(pipeline_log, n)
        if p_tail:
            sections.append("=== pipeline.log ===")
            sections.append(p_tail)

    return PlainTextResponse("\n\n".join(sections))


@app.get("/api/job/{job_id}/outputs")
def job_outputs(job_id: str) -> dict[str, Any]:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_str = st.get("run_dir")
    if not run_dir_str:
        return {"files": []}

    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        return {"files": []}

    files: set[str] = set()
    for p in run_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(run_dir)).replace("\\", "/")
            files.add(rel)
            # Expose a virtual uncompressed path for *.gz artifacts.
            if rel.endswith(".gz"):
                files.add(rel[:-3])

    return {"files": sorted(files)}


@app.get("/api/job/{job_id}/bundle")
def job_bundle(job_id: str):
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail="Job is not complete yet.")

    bundle = Path(st.get("bundle_path") or "")
    if not bundle.exists():
        run_dir_str = st.get("run_dir")
        if not run_dir_str:
            raise HTTPException(status_code=404, detail="Bundle not found")
        bundle = Path(run_dir_str) / "bundle" / "results.zip"
        if not bundle.exists():
            raise HTTPException(status_code=404, detail="Bundle not found")

    return FileResponse(bundle, filename=f"{job_id}_results.zip")


@app.get("/api/job/{job_id}/download")
def job_download(job_id: str, path: str = Query(..., min_length=1)):
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_str = st.get("run_dir")
    if not run_dir_str:
        raise HTTPException(status_code=409, detail="Run directory is not available yet.")

    run_dir = Path(run_dir_str).resolve()
    target = (run_dir / path).resolve()
    if run_dir not in target.parents and target != run_dir:
        raise HTTPException(status_code=400, detail="Invalid path")
    if target.exists() and target.is_file():
        return FileResponse(target, filename=target.name)

    gz_target = Path(str(target) + ".gz")
    if run_dir not in gz_target.parents and gz_target != run_dir:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not gz_target.exists() or not gz_target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{target.name}"'}
    return StreamingResponse(_iter_gzip_bytes(gz_target), media_type=media_type, headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
