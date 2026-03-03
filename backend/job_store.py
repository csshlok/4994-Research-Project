from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "runner.log"

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def create_job(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            now = time.time()
            rec = {
                "job_id": job_id,
                "created_at": now,
                "updated_at": now,
            }
            rec.update(payload)
            self._write_json(self.status_path(job_id), rec)

    def update_status(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            path = self.status_path(job_id)
            rec = self._read_json(path)
            if not rec:
                rec = {
                    "job_id": job_id,
                    "created_at": time.time(),
                }
            rec.update(fields)
            rec["updated_at"] = time.time()
            self._write_json(path, rec)

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        path = self.status_path(job_id)
        if not path.exists():
            return None
        return self._read_json(path)

    def append_log(self, job_id: str, line: str) -> None:
        lp = self.log_path(job_id)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with lp.open("a", encoding="utf-8", errors="replace") as f:
                f.write(line.rstrip("\n") + "\n")

    def tail_log(self, job_id: str, n: int = 200) -> str:
        lp = self.log_path(job_id)
        if not lp.exists():
            return ""
        lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max(1, int(n)):])

