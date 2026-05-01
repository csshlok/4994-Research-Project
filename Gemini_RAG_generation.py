from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
DEFAULT_SCORE_ROOT = ROOT / "company scores"
DEFAULT_MANIFEST = DEFAULT_SCORE_ROOT / "manifest.json"
DEFAULT_ENV_FILE = ROOT / ".env.render"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash-lite"


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "key_strengths": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "domain_explanations": {
            "type": "object",
            "properties": {
                "strongest": {"type": "string"},
                "weakest": {"type": "string"},
            },
            "required": ["strongest", "weakest"],
        },
        "what_stands_out": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "cluster_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string"},
                    "label": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["cluster_id", "label", "summary", "evidence_ids"],
            },
        },
    },
    "required": [
        "executive_summary",
        "key_strengths",
        "key_risks",
        "domain_explanations",
        "what_stands_out",
        "cluster_summaries",
    ],
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def manifest_companies(score_root: Path, manifest_path: Path, requested: list[str] | None) -> list[str]:
    if requested:
        wanted = {company.casefold(): company for company in requested}
        by_dir = {path.name.casefold(): path.name for path in score_root.iterdir() if path.is_dir()}
        missing = [company for key, company in wanted.items() if key not in by_dir]
        if missing:
            raise FileNotFoundError(f"Company score folder(s) not found: {', '.join(missing)}")
        return [by_dir[key] for key in wanted]
    if manifest_path.exists():
        payload = load_json(manifest_path)
        companies = [str(company) for company in payload.get("companies", [])]
        if companies:
            return companies
    return [path.name for path in sorted(score_root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]


def compact_cluster(cluster: dict[str, Any], evidence_limit: int) -> dict[str, Any]:
    evidence = []
    for item in cluster.get("evidence", [])[:evidence_limit]:
        evidence.append(
            {
                "evidence_id": item.get("evidence_id", ""),
                "rating": item.get("rating", ""),
                "role": item.get("role", ""),
                "text": item.get("text", ""),
            }
        )
    return {
        "cluster_id": cluster.get("cluster_id", ""),
        "mode": cluster.get("mode", ""),
        "label": cluster.get("label", ""),
        "domain": cluster.get("domain", ""),
        "review_count": cluster.get("review_count", 0),
        "signal": cluster.get("signal", 0),
        "terms": cluster.get("terms", []),
        "evidence": evidence,
    }


def build_prompt(company: str, evidence_payload: dict[str, Any], evidence_limit: int) -> str:
    clusters = evidence_payload.get("clusters", [])
    selected_clusters = sorted(
        clusters,
        key=lambda cluster: (str(cluster.get("mode", "")), -int(cluster.get("review_count", 0))),
    )
    compact_payload = {
        "company": company,
        "score_summary": evidence_payload.get("score_summary", {}),
        "clusters": [compact_cluster(cluster, evidence_limit) for cluster in selected_clusters],
    }
    return (
        "You are writing RAG-backed workplace analytics from employee review evidence.\n"
        "Use only the supplied data. Do not invent causes, policies, or facts.\n"
        "Do not quote employee text verbatim; paraphrase patterns in plain language.\n"
        "Write for a normal reader with no background in the scoring science.\n"
        "The executive_summary must be exactly two professional paragraphs.\n"
        "Mention that the domains are signals from review language, not direct proof of every employee's experience.\n"
        "Keep every sentence concise and evidence-grounded.\n\n"
        f"Input JSON:\n{json.dumps(compact_payload, ensure_ascii=False)}"
    )


def parse_response_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def usage_payload(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {}
    if hasattr(metadata, "model_dump"):
        return metadata.model_dump(mode="json", exclude_none=True)
    return json.loads(json.dumps(metadata, default=str))


def call_gemini(
    client: genai.Client,
    models: list[str],
    prompt: str,
    temperature: float,
    retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    attempts: list[dict[str, str]] = []
    for model in models:
        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                        response_json_schema=RESPONSE_SCHEMA,
                    ),
                )
                return parse_response_text(response.text or "{}"), {
                    "model": model,
                    "usage": usage_payload(response),
                    "attempts": attempts,
                }
            except Exception as exc:
                last_error = exc
                attempts.append({"model": model, "attempt": str(attempt + 1), "error": str(exc)})
                if attempt < retries:
                    time.sleep(retry_sleep * (attempt + 1))
                    continue
                break
    raise RuntimeError(f"Gemini generation failed: {last_error}; attempts={attempts}")


def split_artifacts(company: str, generated: dict[str, Any], meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    common_meta = {
        "schema_version": 1,
        "company": company,
        "generated_at": now_iso(),
        "generator": "gemini",
        **meta,
    }
    return {
        "rag_summary.json": {
            **common_meta,
            "executive_summary": generated.get("executive_summary", []),
            "key_strengths": generated.get("key_strengths", []),
            "key_risks": generated.get("key_risks", []),
            "domain_explanations": generated.get("domain_explanations", {}),
        },
        "rag_clusters.json": {
            **common_meta,
            "cluster_summaries": generated.get("cluster_summaries", []),
        },
        "rag_insights.json": {
            **common_meta,
            "what_stands_out": generated.get("what_stands_out", []),
        },
    }


def generate_company(
    client: genai.Client,
    company_dir: Path,
    models: list[str],
    temperature: float,
    retries: int,
    retry_sleep: float,
    evidence_limit: int,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    output_paths = [company_dir / "rag_summary.json", company_dir / "rag_clusters.json", company_dir / "rag_insights.json"]
    if not force and all(path.exists() for path in output_paths):
        return {"company": company_dir.name, "status": "cached"}

    evidence_path = company_dir / "rag_evidence.json"
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing {evidence_path}")
    evidence_payload = load_json(evidence_path)
    prompt = build_prompt(company_dir.name, evidence_payload, evidence_limit)
    if dry_run:
        return {"company": company_dir.name, "status": "dry_run", "prompt_chars": len(prompt)}

    generated, meta = call_gemini(client, models, prompt, temperature, retries, retry_sleep)
    artifacts = split_artifacts(company_dir.name, generated, meta)
    for filename, payload in artifacts.items():
        write_json(company_dir / filename, payload)
    return {
        "company": company_dir.name,
        "status": "generated",
        "model": meta.get("model"),
        "prompt_tokens": meta.get("usage", {}).get("prompt_token_count"),
        "output_tokens": meta.get("usage", {}).get("candidates_token_count"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gemini-backed cached RAG artifacts from rag_evidence.json.")
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--company", action="append", default=None)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--fallback-model", default=os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL))
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=4.0)
    parser.add_argument("--evidence-limit", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    if not os.environ.get("GEMINI_API_KEY") and not args.dry_run:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    score_root = Path(args.score_root)
    companies = manifest_companies(score_root, Path(args.manifest), args.company)
    models = [args.model]
    if not args.no_fallback and args.fallback_model and args.fallback_model not in models:
        models.append(args.fallback_model)
    client = genai.Client() if not args.dry_run else None

    results = []
    skipped = []
    for company in companies:
        try:
            results.append(
                generate_company(
                    client=client,
                    company_dir=score_root / company,
                    models=models,
                    temperature=args.temperature,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    evidence_limit=args.evidence_limit,
                    force=args.force,
                    dry_run=args.dry_run,
                )
            )
        except Exception as exc:
            skipped.append({"company": company, "reason": str(exc)})
            if args.company:
                break

    print(json.dumps({"generated": len([r for r in results if r.get("status") == "generated"]), "results": results, "skipped": skipped}, indent=2))
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
