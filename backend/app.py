from __future__ import annotations

import json
import threading
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
from fastapi.responses import FileResponse, PlainTextResponse
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


def _read_progress_snapshot(progress_path: Path) -> dict[str, Any]:
    if not progress_path.exists():
        return {"current_stage": None, "current_status": None, "history": []}

    history: list[dict[str, Any]] = []
    for line in progress_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, n):])


app = FastAPI(title="Pipeline Backend", version="1.0.0")
store = JobStore(settings.JOBS_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "repo_root": str(settings.REPO_ROOT),
        "review_data_dir": str(settings.REVIEW_DATA_DIR),
        "runs_dir": str(settings.RUNS_DIR),
        "cache_usage_log": str(settings.CACHE_USAGE_LOG),
        "python_exe": str(settings.PYTHON_EXE),
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

    _spawn(run_cache_job, store, job_id, company)
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
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                st["pipeline_report"] = {
                    "counts": report.get("counts"),
                    "errors": report.get("errors"),
                    "stages": report.get("stages"),
                }
            except Exception:
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

    files = []
    for p in run_dir.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(run_dir)).replace("\\", "/"))

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
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(target, filename=target.name)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
