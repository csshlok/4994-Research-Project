from __future__ import annotations

import gzip
import hmac
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
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


class CompareRagRequest(BaseModel):
    companies: list[str]


class MatchProfileRequest(BaseModel):
    narrative: str | None = None
    tradeoffs: list[str] = []
    accepted_tradeoffs: list[str] = []
    local_profile: dict[str, Any] | None = None


class MatchTopSummaryRequest(BaseModel):
    profile: dict[str, Any]
    top_match: dict[str, Any]
    evidence: list[dict[str, Any]] = []


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


def _company_match_key(text: str) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def _safe_slug(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text).strip())
    return s.strip("_").lower() or "company"


def _load_env_file_if_needed() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = settings.REPO_ROOT / ".env.render"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


def _google_genai_available() -> bool:
    try:
        from google import genai  # noqa: F401
        from google.genai import types  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_scored_company_dir(company_id: str) -> Path | None:
    root = settings.COMPANY_SCORES_DIR
    wanted = str(company_id or "").strip()
    if not wanted or not root.exists():
        return None

    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None

    for p in dirs:
        if p.name == wanted:
            return p

    wanted_lower = wanted.lower()
    wanted_slug = _safe_slug(wanted)
    wanted_key = _company_match_key(wanted)
    for p in dirs:
        if p.name.lower() == wanted_lower:
            return p
    for p in dirs:
        if _safe_slug(p.name) == wanted_slug:
            return p
    for p in dirs:
        if _company_match_key(p.name) == wanted_key:
            return p
    return None


def _read_company_csv_score_summary(company_dir: Path) -> dict[str, Any]:
    text = _read_text_maybe_gz(company_dir / "company_scores.csv")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    import csv
    from io import StringIO

    rows = list(csv.DictReader(StringIO("\n".join(lines))))
    if not rows:
        return {}
    row = rows[0]

    def num(name: str) -> float:
        try:
            return float(row.get(name) or 0)
        except Exception:
            return 0.0

    domains = {
        "Physiological": round((num("G_smoothed_final_phys") + 1) * 50),
        "Self-Protection": round((num("G_smoothed_final_selfprot") + 1) * 50),
        "Affiliation": round((num("G_smoothed_final_aff") + 1) * 50),
        "Status & Esteem": round((num("G_smoothed_final_stat") + 1) * 50),
        "Family Care": round((num("G_smoothed_final_fam") + 1) * 50),
    }
    return {
        "review_count": round(num("n_reviews")),
        "overall_score": round(sum(domains.values()) / max(1, len(domains))),
        "positive_share": num("pos_share"),
        "negative_share": num("neg_share"),
        "domain_scores": domains,
    }


def _load_review_texts_by_id(path: Path) -> dict[str, str]:
    text = _read_text_maybe_gz(path)
    if not text:
        return {}

    import csv
    from io import StringIO

    try:
        rows = list(csv.DictReader(StringIO(text)))
    except Exception:
        return {}

    out: dict[str, str] = {}
    for row in rows:
        normalized_row = {
            str(key or "").lstrip("\ufeff").strip().strip('"'): value
            for key, value in row.items()
        }
        review_id = str(normalized_row.get("review_id") or normalized_row.get("id") or "").strip()
        if review_id.endswith(".0"):
            review_id = review_id[:-2]
        if not review_id:
            continue
        parts = [
            normalized_row.get("pros") or normalized_row.get("pros_text") or "",
            normalized_row.get("cons") or normalized_row.get("cons_text") or "",
            normalized_row.get("body") or "",
            normalized_row.get("summary") or "",
        ]
        combined = _normalize_review_text(" ".join(str(part).strip() for part in parts if str(part).strip()))
        if combined:
            out[review_id] = combined
    return out


def _normalize_review_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "â": "'",
        "â": "'",
        "â": '"',
        "â": '"',
        "â": "-",
        "â": "-",
        "â¢": " ",
        "Ã¢ÂÂ": "'",
        "Ã¢ÂÂ": "'",
        "Ã¢ÂÂ": '"',
        "Ã¢ÂÂ": '"',
        "Ã¢ÂÂ": "-",
        "Ã¢ÂÂ": "-",
        "Ã¢ÂÂ¢": " ",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
        "\u2022": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"[*#_`~]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _enrich_rag_evidence_text(evidence: dict[str, Any] | None, company_dir: Path) -> dict[str, Any] | None:
    if not isinstance(evidence, dict):
        return evidence

    source_files = evidence.get("source_files")
    reviews_path: Path | None = None
    if isinstance(source_files, dict) and isinstance(source_files.get("reviews"), str):
        candidate = (settings.REPO_ROOT / source_files["reviews"]).resolve()
        if candidate.exists():
            reviews_path = candidate
    if reviews_path is None:
        candidate = settings.REVIEW_DATA_DIR / company_dir.name / "reviews.csv"
        if candidate.exists():
            reviews_path = candidate

    if reviews_path is None:
        return evidence

    review_texts = _load_review_texts_by_id(reviews_path)
    if not review_texts:
        return evidence

    def enrich_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        review_id = str(item.get("review_id") or "").strip()
        if review_id.endswith(".0"):
            review_id = review_id[:-2]
        full_text = review_texts.get(review_id)
        if full_text:
            item["text"] = full_text
        elif item.get("text"):
            item["text"] = _normalize_review_text(item["text"])

    for cluster in evidence.get("clusters", []) if isinstance(evidence.get("clusters"), list) else []:
        if not isinstance(cluster, dict):
            continue
        for item in cluster.get("evidence", []) if isinstance(cluster.get("evidence"), list) else []:
            enrich_item(item)

    for item in evidence.get("flat_evidence", []) if isinstance(evidence.get("flat_evidence"), list) else []:
        enrich_item(item)

    return evidence


def _has_required_score_files(company_dir: Path) -> bool:
    return (company_dir / "review_scores.csv").exists() and (company_dir / "company_scores.csv").exists()


ACTIVE_JOB_STATUSES = {"queued", "running"}
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _require_api_token(authorization: str | None = Header(default=None)) -> None:
    expected = settings.PIPELINE_API_TOKEN
    if not expected:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token",
        )


def _validate_job_id(job_id: str) -> str:
    if not JOB_ID_RE.fullmatch(str(job_id or "")):
        raise HTTPException(status_code=400, detail="Invalid job id")
    return job_id


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _trusted_run_dir(raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    run_dir = Path(raw).resolve()
    if not _is_relative_to(run_dir, settings.RUNS_DIR):
        raise HTTPException(status_code=400, detail="Invalid run directory")
    return run_dir


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

    ttl_seconds = float(settings.JOB_RETENTION_SECONDS)
    run_ttl_seconds = float(settings.RUN_RETENTION_SECONDS)
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue

        status_path = job_dir / "status.json"
        status = _load_status_file(status_path)
        state = str(status.get("status") or "").strip().lower()
        run_dir_val = status.get("run_dir")
        if state in ACTIVE_JOB_STATUSES and isinstance(run_dir_val, str) and run_dir_val.strip():
            run_path = Path(run_dir_val).resolve()
            if _is_relative_to(run_path, settings.RUNS_DIR):
                active_runs.add(run_path)

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
            if isinstance(run_dir_val, str) and run_dir_val.strip():
                try:
                    run_path = Path(run_dir_val).resolve()
                    if (
                        _is_relative_to(run_path, settings.RUNS_DIR)
                        and run_path.exists()
                        and run_path.is_dir()
                        and (now_ts - age_ref) >= run_ttl_seconds
                    ):
                        shutil.rmtree(run_path, ignore_errors=True)
                except Exception:
                    pass
            shutil.rmtree(job_dir, ignore_errors=True)

    return active_runs


def _cleanup_runs(now_ts: float, active_runs: set[Path]) -> None:
    runs_root = settings.RUNS_DIR
    if not runs_root.exists():
        return

    ttl_seconds = float(settings.RUN_RETENTION_SECONDS)
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        run_resolved = run_dir.resolve()
        if run_resolved in active_runs:
            continue
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    counts = _count_jobs_by_status()
    _load_env_file_if_needed()
    return {
        "ok": True,
        "service": "pipeline-backend",
        "gemini_api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "google_genai_available": _google_genai_available(),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        "gemini_fallback_model": os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
        "run_retention_seconds": settings.RUN_RETENTION_SECONDS,
        "job_retention_seconds": settings.JOB_RETENTION_SECONDS,
        "cleanup_interval_seconds": settings.CLEANUP_INTERVAL_SECONDS,
        "job_dispatch_interval_seconds": settings.JOB_DISPATCH_INTERVAL_SECONDS,
        "max_queue_length": settings.MAX_QUEUE_LENGTH,
        "queue_depth": counts["queued"],
        "running_jobs": counts["running"],
        "company_scores_dir": str(settings.COMPANY_SCORES_DIR),
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


@app.get("/api/scored-companies")
def list_scored_companies() -> dict[str, Any]:
    root = settings.COMPANY_SCORES_DIR
    if not root.exists():
        return {"companies": []}

    companies: list[dict[str, Any]] = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
        has_scores = _has_required_score_files(p)
        if not has_scores:
            continue
        companies.append(
            {
                "id": p.name,
                "has_review_scores": (p / "review_scores.csv").exists(),
                "has_company_scores": (p / "company_scores.csv").exists(),
                "has_cleaned_reviews": (p / "cleaned_reviews.csv").exists(),
                "has_topics": (p / "topic_summary.csv").exists(),
                "has_rag": (p / "rag_summary.json").exists()
                and (p / "rag_clusters.json").exists()
                and (p / "rag_insights.json").exists(),
            }
        )
    return {"companies": companies}


@app.get("/api/scored-company/{company_id}/outputs", dependencies=[Depends(_require_api_token)])
def scored_company_outputs(company_id: str) -> dict[str, Any]:
    company_dir = _resolve_scored_company_dir(company_id)
    if company_dir is None or not _has_required_score_files(company_dir):
        raise HTTPException(status_code=404, detail="Scored company not found")

    files: set[str] = set()
    for p in company_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(company_dir)).replace("\\", "/")
            files.add(rel)
            if rel.endswith(".gz"):
                files.add(rel[:-3])
    return {"company_id": company_dir.name, "files": sorted(files)}


@app.get("/api/scored-company/{company_id}/rag", dependencies=[Depends(_require_api_token)])
def scored_company_rag(company_id: str) -> dict[str, Any]:
    company_dir = _resolve_scored_company_dir(company_id)
    if company_dir is None or not _has_required_score_files(company_dir):
        raise HTTPException(status_code=404, detail="Scored company not found")

    summary = _read_json_maybe_gz(company_dir / "rag_summary.json")
    clusters = _read_json_maybe_gz(company_dir / "rag_clusters.json")
    insights = _read_json_maybe_gz(company_dir / "rag_insights.json")
    evidence = _read_json_maybe_gz(company_dir / "rag_evidence.json")
    evidence = _enrich_rag_evidence_text(evidence, company_dir)
    profile = _read_json_maybe_gz(company_dir / "rag_profile.json")
    if not summary and not clusters and not insights:
        raise HTTPException(status_code=404, detail="RAG artifacts not found")

    return {
        "company_id": company_dir.name,
        "summary": summary,
        "clusters": clusters,
        "insights": insights,
        "evidence": evidence,
        "profile": profile,
    }


COMPARE_RAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "key_differences": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "shared_risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "company_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "strength": {"type": "string"},
                    "risk": {"type": "string"},
                },
                "required": ["company", "strength", "risk"],
            },
        },
        "best_fit_by_need": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "need": {"type": "string"},
                    "company": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["need", "company", "reason"],
            },
        },
    },
    "required": [
        "executive_summary",
        "key_differences",
        "shared_risks",
        "company_notes",
        "best_fit_by_need",
    ],
}


MATCH_PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "likely_motivators": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "core_needs": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "risk_sensitivities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "work_style": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "summary",
        "likely_motivators",
        "core_needs",
        "risk_sensitivities",
        "work_style",
        "confidence",
    ],
}


MATCH_TOP_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "match_reasons": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "evidence_connections": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "caveat": {"type": "string"},
    },
    "required": ["headline", "summary", "match_reasons", "evidence_connections", "caveat"],
}


def _comparison_cache_dir(company_dirs: list[Path]) -> Path:
    key = "-".join(sorted(_safe_slug(path.name) for path in company_dirs))
    return settings.REPO_ROOT / "local_compare_outputs" / "comparison_rag" / key


def _top_cluster_labels(payload: dict[str, Any] | None, limit: int = 3) -> list[str]:
    clusters = payload.get("cluster_summaries", []) if isinstance(payload, dict) else []
    if not isinstance(clusters, list):
        return []
    labels: list[str] = []
    for item in clusters[:limit]:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            labels.append(item["label"])
    return labels


def _build_compare_payload(company_dirs: list[Path]) -> dict[str, Any]:
    companies: list[dict[str, Any]] = []
    for company_dir in company_dirs:
        summary = _read_json_maybe_gz(company_dir / "rag_summary.json") or {}
        insights = _read_json_maybe_gz(company_dir / "rag_insights.json") or {}
        clusters = _read_json_maybe_gz(company_dir / "rag_clusters.json") or {}
        profile = _read_json_maybe_gz(company_dir / "rag_profile.json") or {}
        score_summary = profile.get("score_summary") if isinstance(profile, dict) else None
        if not isinstance(score_summary, dict):
            score_summary = _read_company_csv_score_summary(company_dir)

        companies.append(
            {
                "company": company_dir.name,
                "score_summary": score_summary,
                "executive_summary": summary.get("executive_summary", []) if isinstance(summary, dict) else [],
                "key_strengths": summary.get("key_strengths", []) if isinstance(summary, dict) else [],
                "key_risks": summary.get("key_risks", []) if isinstance(summary, dict) else [],
                "what_stands_out": insights.get("what_stands_out", []) if isinstance(insights, dict) else [],
                "cluster_labels": _top_cluster_labels(clusters),
            }
        )
    return {"companies": companies}


def _fallback_compare_rag(company_dirs: list[Path], error: str | None = None) -> dict[str, Any]:
    payload = _build_compare_payload(company_dirs)
    companies = payload["companies"]
    domain_rows: dict[str, list[tuple[str, int]]] = {}
    for company in companies:
        scores = company.get("score_summary", {}).get("domain_scores", {})
        if not isinstance(scores, dict):
            continue
        for domain, score in scores.items():
            try:
                domain_rows.setdefault(str(domain), []).append((str(company["company"]), int(score)))
            except Exception:
                continue

    differences: list[str] = []
    for domain, rows in domain_rows.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda item: item[1], reverse=True)
        differences.append(f"{rows[0][0]} leads {rows[-1][0]} in {domain} by {rows[0][1] - rows[-1][1]} points.")
    differences = differences[:4]

    company_names = ", ".join(str(company["company"]) for company in companies)
    return {
        "schema_version": 1,
        "source": "fallback",
        "error": error,
        "executive_summary": [
            f"This comparison uses cached company scores and RAG profiles for {company_names}. The model-generated comparison was unavailable, so this summary is based on deterministic score gaps and cached single-company signals.",
            "Use the heatmap and company detail cards to inspect where each company is strongest or weakest. Larger domain gaps indicate clearer differences in how employees describe their workplace needs.",
        ],
        "key_differences": differences or ["The selected companies have similar final goal profiles across the compared domains."],
        "shared_risks": [
            "Review-derived signals should be interpreted as aggregated language patterns, not as proof of every employee's experience."
        ],
        "company_notes": [
            {
                "company": str(company["company"]),
                "strength": "; ".join(company.get("key_strengths", [])[:1]) or "See cached company summary for the strongest signals.",
                "risk": "; ".join(company.get("key_risks", [])[:1]) or "See cached company summary for the main risks.",
            }
            for company in companies
        ],
        "best_fit_by_need": [],
    }


def _call_gemini_compare(prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _load_env_file_if_needed()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(f"google-genai is not installed: {exc}") from exc

    models = [
        os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
    ]
    models = [model for index, model in enumerate(models) if model and model not in models[:index]]
    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None
    attempts: list[dict[str, str]] = []
    for model in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json",
                        response_json_schema=COMPARE_RAG_SCHEMA,
                    ),
                )
                text = response.text or "{}"
                generated = json.loads(text.strip().strip("`").removeprefix("json").strip())
                usage = getattr(response, "usage_metadata", None)
                if hasattr(usage, "model_dump"):
                    usage_payload = usage.model_dump(mode="json", exclude_none=True)
                else:
                    usage_payload = {}
                return generated, {"model": model, "usage": usage_payload, "attempts": attempts}
            except Exception as exc:
                last_error = exc
                attempts.append({"model": model, "attempt": str(attempt + 1), "error": str(exc)})
                if attempt < 2:
                    time.sleep(4 * (attempt + 1))
        continue
    raise RuntimeError(f"Gemini comparison generation failed: {last_error}; attempts={attempts}")


def _call_gemini_match_profile(prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _load_env_file_if_needed()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(f"google-genai is not installed: {exc}") from exc

    models = [
        os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
    ]
    models = [model for index, model in enumerate(models) if model and model not in models[:index]]
    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None
    attempts: list[dict[str, str]] = []
    for model in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.35,
                        response_mime_type="application/json",
                        response_json_schema=MATCH_PROFILE_SCHEMA,
                    ),
                )
                text = response.text or "{}"
                generated = json.loads(text.strip().strip("`").removeprefix("json").strip())
                usage = getattr(response, "usage_metadata", None)
                if hasattr(usage, "model_dump"):
                    usage_payload = usage.model_dump(mode="json", exclude_none=True)
                else:
                    usage_payload = {}
                return generated, {"model": model, "usage": usage_payload, "attempts": attempts}
            except Exception as exc:
                last_error = exc
                attempts.append({"model": model, "attempt": str(attempt + 1), "error": str(exc)})
                if attempt < 2:
                    time.sleep(4 * (attempt + 1))
        continue
    raise RuntimeError(f"Gemini match profile generation failed: {last_error}; attempts={attempts}")


def _call_gemini_match_top_summary(prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _load_env_file_if_needed()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(f"google-genai is not installed: {exc}") from exc

    models = [
        os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
    ]
    models = [model for index, model in enumerate(models) if model and model not in models[:index]]
    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None
    attempts: list[dict[str, str]] = []
    for model in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.25,
                        response_mime_type="application/json",
                        response_json_schema=MATCH_TOP_SUMMARY_SCHEMA,
                    ),
                )
                text = response.text or "{}"
                generated = json.loads(text.strip().strip("`").removeprefix("json").strip())
                usage = getattr(response, "usage_metadata", None)
                if hasattr(usage, "model_dump"):
                    usage_payload = usage.model_dump(mode="json", exclude_none=True)
                else:
                    usage_payload = {}
                return generated, {"model": model, "usage": usage_payload, "attempts": attempts}
            except Exception as exc:
                last_error = exc
                attempts.append({"model": model, "attempt": str(attempt + 1), "error": str(exc)})
                if attempt < 2:
                    time.sleep(4 * (attempt + 1))
        continue
    raise RuntimeError(f"Gemini match top-summary generation failed: {last_error}; attempts={attempts}")


def _fallback_match_profile(payload: MatchProfileRequest, error: str | None = None) -> dict[str, Any]:
    local = payload.local_profile or {}
    behavior = local.get("behaviorProfile") if isinstance(local, dict) else None
    if not isinstance(behavior, dict):
        behavior = {}

    desired = local.get("desiredThemes", []) if isinstance(local, dict) else []
    avoided = local.get("avoidThemes", []) if isinstance(local, dict) else []
    weights = local.get("weights", {}) if isinstance(local, dict) else {}
    core_needs = [
        key
        for key, _value in sorted(
            ((str(k), float(v)) for k, v in weights.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
    ][:3]

    return {
        "schema_version": 1,
        "source": "fallback",
        "error": error,
        "summary": behavior.get("summary")
        or "Your profile is based on the tradeoffs and workplace language you provided. It emphasizes the conditions you want supplied by an employer and the risks you want to avoid.",
        "likely_motivators": desired[:5] or ["clear support", "workplace fit"],
        "core_needs": core_needs or behavior.get("coreNeeds") or ["self-protection", "affiliation"],
        "risk_sensitivities": avoided[:5] or ["poor fit between needs and environment"],
        "work_style": behavior.get("workStyle")
        or "You appear to evaluate employers by the day-to-day environment they create, not only by brand or compensation.",
        "confidence": behavior.get("confidence") or (0.55 if not (payload.narrative or "").strip() else 0.7),
    }


def _fallback_match_top_summary(payload: MatchTopSummaryRequest, error: str | None = None) -> dict[str, Any]:
    top = payload.top_match or {}
    company = str(top.get("company") or top.get("label") or "the top match")
    score = top.get("score")
    evidence = [
        str(item.get("text") or item.get("label") or "").strip()
        for item in payload.evidence
        if isinstance(item, dict) and str(item.get("text") or item.get("label") or "").strip()
    ][:3]

    return {
        "schema_version": 1,
        "source": "fallback",
        "error": error,
        "headline": f"{company} is currently the strongest match",
        "summary": f"{company} ranks first because its review-derived need scores align best with the user's weighted workplace profile. Its match score is {score}%." if score is not None else f"{company} ranks first because its review-derived need scores align best with the user's weighted workplace profile.",
        "match_reasons": [
            "The company performs relatively well on the needs that the profile weights most heavily.",
            "The available evidence gives the match enough review support to make it worth inspecting first.",
        ],
        "evidence_connections": evidence or ["Review evidence should be inspected below to validate the fit."],
        "caveat": "This is a behavioral fit signal based on review language, not a guarantee of team-level experience.",
    }


@app.post("/api/match/profile", dependencies=[Depends(_require_api_token)])
def match_profile(payload: MatchProfileRequest) -> dict[str, Any]:
    if not (payload.narrative or "").strip() and not payload.tradeoffs and not payload.accepted_tradeoffs:
        raise HTTPException(status_code=400, detail="Provide narrative text or at least one tradeoff.")

    prompt_payload = {
        "narrative": payload.narrative or "",
        "tradeoffs": payload.tradeoffs,
        "accepted_tradeoffs": payload.accepted_tradeoffs,
        "local_profile": payload.local_profile or {},
    }
    prompt = (
        "You are interpreting a workplace-fit profile for an employee-to-company matching tool.\n"
        "Use behavioral science language, but do not diagnose personality or mental health.\n"
        "Write directly to the user in second person using 'you' and 'your'.\n"
        "Ground the interpretation in needs-supplies fit, risk sensitivity, motivation, and workplace tradeoffs.\n"
        "The five workplace domains are physiological support, self-protection, affiliation, status/esteem, and family-care.\n"
        "Do not rank companies. Only explain the user's likely workplace needs and risk sensitivities.\n"
        "If narrative text is blank, infer only from selected tradeoffs and the local profile.\n"
        "Return concise JSON matching the schema.\n\n"
        f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
    )

    try:
        generated, meta = _call_gemini_match_profile(prompt)
        return {
            "schema_version": 1,
            "source": "gemini",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **meta,
            **generated,
        }
    except Exception as exc:
        return _fallback_match_profile(payload, error=str(exc))


@app.post("/api/match/top-summary", dependencies=[Depends(_require_api_token)])
def match_top_summary(payload: MatchTopSummaryRequest) -> dict[str, Any]:
    prompt_payload = {
        "profile": payload.profile,
        "top_match": payload.top_match,
        "evidence": payload.evidence[:6],
    }
    prompt = (
        "You are writing a top-match explanation for an employee-to-company behavioral matching product.\n"
        "Explain why the top company is the best first match for this user using needs-supplies fit, profile priorities, match score components, and review evidence.\n"
        "Write for a normal user. Do not overclaim. Do not diagnose personality. Do not mention ATS, resumes, or hiring eligibility.\n"
        "The explanation should make the result feel evidence-based and inspectable.\n"
        "Return concise JSON matching the schema.\n\n"
        f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
    )

    try:
        generated, meta = _call_gemini_match_top_summary(prompt)
        return {
            "schema_version": 1,
            "source": "gemini",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **meta,
            **generated,
        }
    except Exception as exc:
        return _fallback_match_top_summary(payload, error=str(exc))


@app.post("/api/compare/rag-summary", dependencies=[Depends(_require_api_token)])
def compare_rag_summary(payload: CompareRagRequest) -> dict[str, Any]:
    raw_companies = [company.strip() for company in payload.companies if company.strip()]
    unique_companies = list(dict.fromkeys(raw_companies))[:3]
    if len(unique_companies) < 2:
        raise HTTPException(status_code=400, detail="Select at least two companies.")

    company_dirs: list[Path] = []
    for company in unique_companies:
        company_dir = _resolve_scored_company_dir(company)
        if company_dir is None or not _has_required_score_files(company_dir):
            raise HTTPException(status_code=404, detail=f"Scored company not found: {company}")
        company_dirs.append(company_dir)

    cache_dir = _comparison_cache_dir(company_dirs)
    cache_path = cache_dir / "rag_comparison.json"
    cached = _read_json_maybe_gz(cache_path)
    if cached:
        return cached

    compare_payload = _build_compare_payload(company_dirs)
    prompt = (
        "You are generating a RAG-backed comparison of employee review analytics.\n"
        "Use only the supplied cached company scores, summaries, risks, strengths, and cluster labels.\n"
        "Do not quote employee reviews verbatim. Do not invent policies, causes, or factual claims.\n"
        "Write for a normal reader. The executive_summary must contain exactly two professional paragraphs.\n"
        "Explain differences across companies and identify which company is strongest for specific workplace needs.\n"
        "Return only valid JSON matching the schema.\n\n"
        f"Input JSON:\n{json.dumps(compare_payload, ensure_ascii=False)}"
    )

    try:
        generated, meta = _call_gemini_compare(prompt)
        result = {
            "schema_version": 1,
            "source": "gemini",
            "company_ids": [path.name for path in company_dirs],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **meta,
            **generated,
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        result = _fallback_compare_rag(company_dirs, error=str(exc))

    return result


@app.get("/api/scored-company/{company_id}/download", dependencies=[Depends(_require_api_token)])
def scored_company_download(company_id: str, path: str = Query(..., min_length=1)):
    company_dir = _resolve_scored_company_dir(company_id)
    if company_dir is None or not _has_required_score_files(company_dir):
        raise HTTPException(status_code=404, detail="Scored company not found")

    root = company_dir.resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if target.exists() and target.is_file():
        return FileResponse(target, filename=target.name)

    gz_target = Path(str(target) + ".gz")
    if root not in gz_target.parents and gz_target != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not gz_target.exists() or not gz_target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{target.name}"'}
    return StreamingResponse(_iter_gzip_bytes(gz_target), media_type=media_type, headers=headers)


@app.post("/api/run", dependencies=[Depends(_require_api_token)])
def run_job(payload: RunRequest) -> dict[str, str]:
    if payload.mode != "cache":
        raise HTTPException(status_code=400, detail="Only mode='cache' is supported right now.")

    company = (payload.company_id or payload.company_name or "").strip()
    if not company:
        raise HTTPException(status_code=400, detail="Provide company_id (or company_name).")

    counts = _count_jobs_by_status()
    if counts["queued"] >= int(settings.MAX_QUEUE_LENGTH):
        raise HTTPException(
            status_code=429,
            detail=f"Queue is full (max queued jobs={settings.MAX_QUEUE_LENGTH}). Please retry later.",
        )

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


@app.get("/api/job/{job_id}", dependencies=[Depends(_require_api_token)])
def job_status(job_id: str) -> dict[str, Any]:
    job_id = _validate_job_id(job_id)
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_str = st.get("run_dir")
    if run_dir_str:
        run_dir = _trusted_run_dir(run_dir_str)
        if run_dir is None:
            raise HTTPException(status_code=409, detail="Run directory is not available yet.")
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


@app.get("/api/job/{job_id}/log", dependencies=[Depends(_require_api_token)])
def job_log(job_id: str, n: int = Query(default=200, ge=1, le=2000)) -> PlainTextResponse:
    job_id = _validate_job_id(job_id)
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
        run_dir = _trusted_run_dir(run_dir_str)
        if run_dir is None:
            raise HTTPException(status_code=409, detail="Run directory is not available yet.")
        pipeline_log = run_dir / "99_logs" / "pipeline.log"
        p_tail = _tail_file(pipeline_log, n)
        if p_tail:
            sections.append("=== pipeline.log ===")
            sections.append(p_tail)

    return PlainTextResponse("\n\n".join(sections))


@app.get("/api/job/{job_id}/outputs", dependencies=[Depends(_require_api_token)])
def job_outputs(job_id: str) -> dict[str, Any]:
    job_id = _validate_job_id(job_id)
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_str = st.get("run_dir")
    if not run_dir_str:
        return {"files": []}

    run_dir = _trusted_run_dir(run_dir_str)
    if run_dir is None:
        return {"files": []}
    if not run_dir.exists():
        return {"files": []}

    files: set[str] = set()
    for p in run_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(run_dir)).replace("\\", "/")
            files.add(rel)
            if rel.endswith(".gz"):
                files.add(rel[:-3])

    return {"files": sorted(files)}


@app.get("/api/job/{job_id}/bundle", dependencies=[Depends(_require_api_token)])
def job_bundle(job_id: str):
    job_id = _validate_job_id(job_id)
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail="Job is not complete yet.")

    bundle_value = st.get("bundle_path")
    bundle = Path(bundle_value).resolve() if isinstance(bundle_value, str) and bundle_value.strip() else None
    if bundle is not None and not _is_relative_to(bundle, settings.RUNS_DIR):
        raise HTTPException(status_code=400, detail="Invalid bundle path")
    if bundle is None or not bundle.exists():
        run_dir_str = st.get("run_dir")
        if not run_dir_str:
            raise HTTPException(status_code=404, detail="Bundle not found")
        run_dir = _trusted_run_dir(run_dir_str)
        if run_dir is None:
            raise HTTPException(status_code=404, detail="Bundle not found")
        bundle = run_dir / "bundle" / "results.zip"
        if not bundle.exists():
            raise HTTPException(status_code=404, detail="Bundle not found")

    return FileResponse(bundle, filename=f"{job_id}_results.zip")


@app.get("/api/job/{job_id}/download", dependencies=[Depends(_require_api_token)])
def job_download(job_id: str, path: str = Query(..., min_length=1)):
    job_id = _validate_job_id(job_id)
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_str = st.get("run_dir")
    if not run_dir_str:
        raise HTTPException(status_code=409, detail="Run directory is not available yet.")

    run_dir = _trusted_run_dir(run_dir_str)
    if run_dir is None:
        raise HTTPException(status_code=409, detail="Run directory is not available yet.")
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
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=8000)
