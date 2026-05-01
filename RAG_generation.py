from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW_ROOT = ROOT / "review data"
DEFAULT_SCORE_ROOT = ROOT / "company scores"
DEFAULT_MANIFEST = DEFAULT_SCORE_ROOT / "manifest.json"

GOALS = {
    "phys": "Physiological",
    "selfprot": "Self-Protection",
    "aff": "Affiliation",
    "stat": "Status & Esteem",
    "fam": "Family Care",
}

DOMAIN_KEYS = {
    "phys": "physiological",
    "selfprot": "self_protection",
    "aff": "affiliation",
    "stat": "status_esteem",
    "fam": "family_care",
}

OUTPUT_FIELDS = {
    "rag_evidence": "rag_evidence.json",
    "rag_profile": "rag_profile.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def normalize_review_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_to_percent(value: Any) -> int:
    return int(round(max(0.0, min(100.0, (as_float(value) + 1.0) * 50.0))))


def compact_text(text: str, limit: int = 420) -> str:
    cleaned = " ".join(str(text or "").replace("\u00a0", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def pick_text(review: dict[str, str], mode: str) -> str:
    title = review.get("title", "")
    body = review.get("body", "")
    pros = review.get("pros", "")
    cons = review.get("cons", "")
    if mode == "fulfillment":
        primary = pros or body or title
        secondary = body if body and body != primary else ""
    elif mode == "hindrance":
        primary = cons or body or title
        secondary = body if body and body != primary else ""
    else:
        primary = body or pros or cons or title
        secondary = ""
    joined = primary if not secondary else f"{primary} {secondary}"
    return compact_text(joined)


def load_manifest_companies(score_root: Path, manifest_path: Path, requested: list[str] | None) -> list[str]:
    if requested:
        wanted = {name.casefold(): name for name in requested}
        by_dir = {p.name.casefold(): p.name for p in score_root.iterdir() if p.is_dir()}
        missing = [name for key, name in wanted.items() if key not in by_dir]
        if missing:
            raise FileNotFoundError(f"Company score folder(s) not found: {', '.join(missing)}")
        return [by_dir[key] for key in wanted]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        companies = [str(name) for name in manifest.get("companies", [])]
        if companies:
            return companies
    return [p.name for p in sorted(score_root.iterdir(), key=lambda item: item.name.lower()) if p.is_dir()]


def review_source_path(review_root: Path, company: str) -> Path:
    company_dir = review_root / company
    csv_path = company_dir / "reviews.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Missing reviews.csv for {company}: {csv_path}")


def company_score_summary(row: dict[str, str]) -> dict[str, Any]:
    domains: dict[str, int] = {}
    raw_domains: dict[str, float] = {}
    for goal, domain in GOALS.items():
        raw = as_float(row.get(f"G_smoothed_final_{goal}", row.get(f"G_mean_final_{goal}", 0)))
        domains[DOMAIN_KEYS[goal]] = score_to_percent(raw)
        raw_domains[DOMAIN_KEYS[goal]] = raw

    strongest_key = max(domains, key=lambda key: domains[key])
    weakest_key = min(domains, key=lambda key: domains[key])
    label_by_key = {DOMAIN_KEYS[goal]: domain for goal, domain in GOALS.items()}
    return {
        "review_count": int(as_float(row.get("n_reviews"), 0)),
        "overall_score": score_to_percent(row.get("S_smoothed", row.get("S_mean", 0))),
        "positive_share": round(as_float(row.get("pos_share"), 0), 4),
        "negative_share": round(as_float(row.get("neg_share"), 0), 4),
        "domain_scores": domains,
        "raw_domain_scores": raw_domains,
        "strongest_domain": {
            "key": strongest_key,
            "label": label_by_key[strongest_key],
            "score": domains[strongest_key],
        },
        "weakest_domain": {
            "key": weakest_key,
            "label": label_by_key[weakest_key],
            "score": domains[weakest_key],
        },
    }


def build_review_lookup(review_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {normalize_review_id(row.get("review_id")): row for row in review_rows if normalize_review_id(row.get("review_id"))}


def build_score_lookup(score_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {normalize_review_id(row.get("review_id")): row for row in score_rows if normalize_review_id(row.get("review_id"))}


def build_cluster_evidence(
    company: str,
    topics: list[dict[str, str]],
    assignments: list[dict[str, str]],
    reviews_by_id: dict[str, dict[str, str]],
    scores_by_id: dict[str, dict[str, str]],
    max_evidence: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_groups: dict[str, list[dict[str, str]]] = {}
    for row in assignments:
        cluster_id = row.get("cluster_id", "")
        if cluster_id:
            assignment_groups.setdefault(cluster_id, []).append(row)

    clusters: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    for topic in topics:
        cluster_id = topic.get("cluster_id", "")
        mode = topic.get("mode", "")
        goal = topic.get("goal", "")
        ranked = sorted(
            assignment_groups.get(cluster_id, []),
            key=lambda row: as_float(row.get("evidence_value")),
            reverse=True,
        )
        evidence_items: list[dict[str, Any]] = []
        for assignment in ranked:
            review_id = normalize_review_id(assignment.get("review_id"))
            review = reviews_by_id.get(review_id)
            score_row = scores_by_id.get(review_id, {})
            if not review:
                continue
            text = pick_text(review, mode)
            if not text:
                continue
            evidence_id = f"{company}:{cluster_id}:{review_id}"
            item = {
                "evidence_id": evidence_id,
                "review_id": review_id,
                "cluster_id": cluster_id,
                "mode": mode,
                "goal": goal,
                "domain": topic.get("domain", GOALS.get(goal, "")),
                "evidence_value": round(as_float(assignment.get("evidence_value")), 4),
                "sentiment_score": round(as_float(score_row.get("S_raw")), 4),
                "rating": review.get("rating", ""),
                "date": review.get("date", ""),
                "role": review.get("role", ""),
                "location": review.get("location", ""),
                "employment_status": review.get("employmentStatus", ""),
                "text": text,
            }
            evidence_items.append(item)
            all_evidence.append(item)
            if len(evidence_items) >= max_evidence:
                break

        terms = [term for term in str(topic.get("terms", "")).split("|") if term]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "mode": mode,
                "label": topic.get("label", ""),
                "domain": topic.get("domain", GOALS.get(goal, "")),
                "goal": goal,
                "review_count": int(as_float(topic.get("review_count"), 0)),
                "signal": int(as_float(topic.get("signal"), 0)),
                "x": int(as_float(topic.get("x"), 0)),
                "y": int(as_float(topic.get("y"), 0)),
                "terms": terms,
                "evidence_ids": [item["evidence_id"] for item in evidence_items],
                "evidence": evidence_items,
            }
        )
    return clusters, all_evidence


def build_company_artifacts(
    company: str,
    review_root: Path,
    score_root: Path,
    max_evidence: int,
) -> dict[str, Any]:
    company_dir = score_root / company
    review_csv = review_source_path(review_root, company)
    company_scores = read_csv(company_dir / "company_scores.csv")
    review_scores = read_csv(company_dir / "review_scores.csv")
    topics = read_csv(company_dir / "topic_summary.csv")
    assignments = read_csv(company_dir / "topic_assignments.csv")
    reviews = read_csv(review_csv)

    if not company_scores:
        raise ValueError(f"No company score row found: {company_dir / 'company_scores.csv'}")

    reviews_by_id = build_review_lookup(reviews)
    scores_by_id = build_score_lookup(review_scores)
    clusters, evidence_items = build_cluster_evidence(
        company=company,
        topics=topics,
        assignments=assignments,
        reviews_by_id=reviews_by_id,
        scores_by_id=scores_by_id,
        max_evidence=max_evidence,
    )
    score_summary = company_score_summary(company_scores[0])
    top_fulfillment = sorted(
        [cluster for cluster in clusters if cluster["mode"] == "fulfillment"],
        key=lambda item: item["review_count"],
        reverse=True,
    )[:3]
    top_hindrance = sorted(
        [cluster for cluster in clusters if cluster["mode"] == "hindrance"],
        key=lambda item: item["review_count"],
        reverse=True,
    )[:3]

    evidence_payload = {
        "schema_version": 1,
        "company": company,
        "source_files": {
            "reviews": display_path(review_csv),
            "company_scores": display_path(company_dir / "company_scores.csv"),
            "review_scores": display_path(company_dir / "review_scores.csv"),
            "topic_summary": display_path(company_dir / "topic_summary.csv"),
            "topic_assignments": display_path(company_dir / "topic_assignments.csv"),
        },
        "score_summary": score_summary,
        "clusters": clusters,
        "evidence": evidence_items,
    }
    profile_payload = {
        "schema_version": 1,
        "company": company,
        "score_summary": score_summary,
        "top_fulfillment_clusters": [
            {key: cluster[key] for key in ["cluster_id", "label", "domain", "goal", "review_count", "signal", "terms", "evidence_ids"]}
            for cluster in top_fulfillment
        ],
        "top_hindrance_clusters": [
            {key: cluster[key] for key in ["cluster_id", "label", "domain", "goal", "review_count", "signal", "terms", "evidence_ids"]}
            for cluster in top_hindrance
        ],
        "evidence_count": len(evidence_items),
    }
    write_json(company_dir / OUTPUT_FIELDS["rag_evidence"], evidence_payload)
    write_json(company_dir / OUTPUT_FIELDS["rag_profile"], profile_payload)
    return {
        "company": company,
        "clusters": len(clusters),
        "evidence": len(evidence_items),
        "strongest": score_summary["strongest_domain"]["label"],
        "weakest": score_summary["weakest_domain"]["label"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready RAG evidence packets from scored company data.")
    parser.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--company", action="append", default=None)
    parser.add_argument("--max-evidence", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_root = Path(args.review_root)
    score_root = Path(args.score_root)
    companies = load_manifest_companies(score_root, Path(args.manifest), args.company)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for company in companies:
        try:
            results.append(build_company_artifacts(company, review_root, score_root, args.max_evidence))
        except Exception as exc:
            skipped.append({"company": company, "reason": str(exc)})

    print(json.dumps({"generated": len(results), "skipped": skipped, "results": results}, indent=2))
    return 1 if skipped and args.company else 0


if __name__ == "__main__":
    raise SystemExit(main())
