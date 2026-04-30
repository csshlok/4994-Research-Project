from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SCORE_ROOT = ROOT / "company scores"
DEFAULT_GOAL_DICT = ROOT / "config" / "goal_dict.json"

GOAL_META = {
    "phys": {"label": "Pay & Benefits", "domain": "Physiological", "x": 82, "y": 45},
    "selfprot": {"label": "Safety & Fairness", "domain": "Self-Protection", "x": 50, "y": 38},
    "aff": {"label": "Culture & Belonging", "domain": "Affiliation", "x": 22, "y": 74},
    "stat": {"label": "Growth & Recognition", "domain": "Status & Esteem", "x": 48, "y": 58},
    "fam": {"label": "Flexibility & Care", "domain": "Family Care", "x": 68, "y": 79},
}

GOAL_DICT_KEYS = {
    "phys": "physiological",
    "selfprot": "self_protection",
    "aff": "affiliation",
    "stat": "status_esteem",
    "fam": "family_care",
}

HINDRANCE_LABELS = {
    "phys": "Burnout & Workload",
    "selfprot": "Unsafe or Unfair Culture",
    "aff": "Disconnected Culture",
    "stat": "Stagnation",
    "fam": "Work-Life Pressure",
}

HINDRANCE_POS = {
    "phys": {"x": 24, "y": 28},
    "selfprot": {"x": 50, "y": 38},
    "aff": {"x": 62, "y": 34},
    "stat": {"x": 70, "y": 24},
    "fam": {"x": 82, "y": 52},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def pct(value: float) -> int:
    return int(round(max(0.0, min(1.0, value)) * 100))


def load_goal_terms(goal_dict_path: Path) -> dict[str, dict[str, list[str]]]:
    raw = json.loads(goal_dict_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[str]]] = {}
    for short, long_key in GOAL_DICT_KEYS.items():
        data = raw.get(long_key, {})
        out[short] = {
            "fulfillment": [str(v).replace("_", " ") for v in data.get("fulfillment", [])[:5]],
            "hindrance": [str(v).replace("_", " ") for v in data.get("hindrance", [])[:5]],
        }
    return out


def build_topics(company_dir: Path, terms: dict[str, dict[str, list[str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_scores = company_dir / "review_scores.csv"
    rows = read_csv(review_scores)
    total = max(1, len(rows))

    topics: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []

    for goal, meta in GOAL_META.items():
        f_col = f"F_raw_{goal}"
        h_col = f"H_raw_{goal}"
        g_col = f"G_final_{goal}"

        f_hits = [row for row in rows if as_float(row.get(f_col)) > 0]
        h_hits = [row for row in rows if as_float(row.get(h_col)) > 0]

        f_signal = pct(len(f_hits) / total)
        h_signal = pct(len(h_hits) / total)

        topics.append(
            {
                "mode": "fulfillment",
                "cluster_id": f"fulfillment_{goal}",
                "label": meta["label"],
                "domain": meta["domain"],
                "goal": goal,
                "review_count": len(f_hits),
                "signal": f_signal,
                "x": meta["x"],
                "y": meta["y"],
                "terms": "|".join(terms.get(goal, {}).get("fulfillment", [])),
            }
        )
        h_pos = HINDRANCE_POS[goal]
        topics.append(
            {
                "mode": "hindrance",
                "cluster_id": f"hindrance_{goal}",
                "label": HINDRANCE_LABELS[goal],
                "domain": meta["domain"],
                "goal": goal,
                "review_count": len(h_hits),
                "signal": h_signal,
                "x": h_pos["x"],
                "y": h_pos["y"],
                "terms": "|".join(terms.get(goal, {}).get("hindrance", [])),
            }
        )

    for row in rows:
        best_mode = "none"
        best_goal = ""
        best_value = 0.0
        for goal in GOAL_META:
            f_val = as_float(row.get(f"F_raw_{goal}"))
            h_val = as_float(row.get(f"H_raw_{goal}"))
            if f_val > best_value:
                best_mode = "fulfillment"
                best_goal = goal
                best_value = f_val
            if h_val > best_value:
                best_mode = "hindrance"
                best_goal = goal
                best_value = h_val
        assignments.append(
            {
                "review_id": row.get("review_id", ""),
                "company_id": row.get("company_id", company_dir.name),
                "mode": best_mode,
                "goal": best_goal,
                "cluster_id": f"{best_mode}_{best_goal}" if best_goal else "",
                "evidence_value": best_value,
            }
        )

    topics.sort(key=lambda r: (str(r["mode"]), -int(r["review_count"])))
    return topics, assignments


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate topic cluster artifacts from precomputed score files.")
    ap.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    ap.add_argument("--goal-dict", default=str(DEFAULT_GOAL_DICT))
    ap.add_argument("--company", action="append", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    score_root = Path(args.score_root)
    terms = load_goal_terms(Path(args.goal_dict))
    companies = sorted([p for p in score_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if args.company:
        wanted = {c.casefold() for c in args.company}
        companies = [p for p in companies if p.name.casefold() in wanted]

    topic_fields = ["mode", "cluster_id", "label", "domain", "goal", "review_count", "signal", "x", "y", "terms"]
    assignment_fields = ["review_id", "company_id", "mode", "goal", "cluster_id", "evidence_value"]

    done = 0
    skipped = 0
    for company_dir in companies:
        if not (company_dir / "review_scores.csv").exists():
            skipped += 1
            continue
        topics, assignments = build_topics(company_dir, terms)
        write_csv(company_dir / "topic_summary.csv", topics, topic_fields)
        write_csv(company_dir / "topic_assignments.csv", assignments, assignment_fields)
        done += 1
    print(f"generated={done} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
