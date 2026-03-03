from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


def _origins_env(name: str, default: str = "*") -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(parts) if parts else ("*",)


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


def load_settings() -> Settings:
    repo_root = _path_env("REPO_ROOT", Path(__file__).resolve().parents[1])
    review_data_dir = _path_env("REVIEW_DATA_DIR", repo_root / "review data")
    goal_dict = _path_env("GOAL_DICT", repo_root / "config" / "goal_dict.json")
    jobs_dir = _path_env("JOBS_DIR", repo_root / "server_jobs")
    cache_usage_log = _path_env("CACHE_USAGE_LOG", jobs_dir / "cache_usage_log.jsonl")
    runs_dir = _path_env("RUNS_DIR", repo_root / "runs")
    python_exe = _path_env("PYTHON_EXE", Path(sys.executable))
    allowed_origins = _origins_env("ALLOWED_ORIGINS", "*")

    return Settings(
        REPO_ROOT=repo_root,
        REVIEW_DATA_DIR=review_data_dir,
        GOAL_DICT=goal_dict,
        JOBS_DIR=jobs_dir,
        CACHE_USAGE_LOG=cache_usage_log,
        RUNS_DIR=runs_dir,
        PYTHON_EXE=python_exe,
        ALLOWED_ORIGINS=allowed_origins,
    )


settings = load_settings()
