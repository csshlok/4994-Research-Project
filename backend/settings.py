from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _origins_env(name: str, default: str = "*") -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(parts) if parts else ("*",)


def _retention_seconds(seconds_name: str, hours_name: str, default_hours: int) -> int:
    raw_seconds = os.environ.get(seconds_name, "").strip()
    if raw_seconds:
        try:
            return max(60, int(raw_seconds))
        except Exception:
            pass
    return max(60, _int_env(hours_name, default_hours) * 3600)


@dataclass(frozen=True)
class Settings:
    REPO_ROOT: Path
    REVIEW_DATA_DIR: Path
    GOAL_DICT: Path
    JOBS_DIR: Path
    CACHE_USAGE_LOG: Path
    RUNS_DIR: Path
    PYTHON_EXE: Path
    ALLOWED_ORIGINS: tuple[str, ...]
    RUN_RETENTION_SECONDS: int
    JOB_RETENTION_SECONDS: int
    CLEANUP_INTERVAL_SECONDS: int
    JOB_DISPATCH_INTERVAL_SECONDS: int
    MAX_QUEUE_LENGTH: int


def load_settings() -> Settings:
    repo_root = _path_env("REPO_ROOT", Path(__file__).resolve().parents[1])
    review_data_dir = _path_env("REVIEW_DATA_DIR", repo_root / "review data")
    goal_dict = _path_env("GOAL_DICT", repo_root / "config" / "goal_dict.json")
    jobs_dir = _path_env("JOBS_DIR", repo_root / "server_jobs")
    cache_usage_log = _path_env("CACHE_USAGE_LOG", jobs_dir / "cache_usage_log.jsonl")
    runs_dir = _path_env("RUNS_DIR", repo_root / "runs")
    python_exe = _path_env("PYTHON_EXE", Path(sys.executable))
    allowed_origins = _origins_env("ALLOWED_ORIGINS", "*")
    run_retention_seconds = _retention_seconds("RUN_RETENTION_SECONDS", "RUN_RETENTION_HOURS", 1)
    job_retention_seconds = _retention_seconds("JOB_RETENTION_SECONDS", "JOB_RETENTION_HOURS", 24)
    cleanup_interval_seconds = max(30, _int_env("CLEANUP_INTERVAL_SECONDS", 600))
    job_dispatch_interval_seconds = max(1, _int_env("JOB_DISPATCH_INTERVAL_SECONDS", 2))
    max_queue_length = max(1, _int_env("MAX_QUEUE_LENGTH", 10))

    return Settings(
        REPO_ROOT=repo_root,
        REVIEW_DATA_DIR=review_data_dir,
        GOAL_DICT=goal_dict,
        JOBS_DIR=jobs_dir,
        CACHE_USAGE_LOG=cache_usage_log,
        RUNS_DIR=runs_dir,
        PYTHON_EXE=python_exe,
        ALLOWED_ORIGINS=allowed_origins,
        RUN_RETENTION_SECONDS=run_retention_seconds,
        JOB_RETENTION_SECONDS=job_retention_seconds,
        CLEANUP_INTERVAL_SECONDS=cleanup_interval_seconds,
        JOB_DISPATCH_INTERVAL_SECONDS=job_dispatch_interval_seconds,
        MAX_QUEUE_LENGTH=max_queue_length,
    )


settings = load_settings()
