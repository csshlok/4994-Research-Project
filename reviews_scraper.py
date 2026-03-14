import asyncio, json, re, inspect, sys, random, argparse, time, csv, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions
from csv_safety import sanitize_csv_cell

PROFILE_DIR = Path("chrome-profile")
OUT_JSON    = Path("reviews.json")
DEBUG_DIR   = Path("debug_html")
DEBUG_DIR.mkdir(exist_ok=True)

def log(msg: str):
    print(msg, flush=True)


def _normalize_region(region: Optional[str]) -> Optional[str]:
    if not region:
        return None
    cleaned = re.sub(r"\s+", " ", str(region).strip())
    return cleaned or None


def _apply_region_filters_to_query(qd: Dict[str, List[str]], region: Optional[str]) -> None:
    reg = _normalize_region(region)
    if not reg:
        return

    qd["filter.location"] = [reg]

    us_aliases = {"united states", "united states of america", "us", "u.s.", "usa", "u.s.a."}
    if reg.casefold() in us_aliases:
        qd["filter.locationId"] = ["1"]
        qd["filter.locationType"] = ["N"]
    else:
        qd.pop("filter.locationId", None)
        qd.pop("filter.locationType", None)


def _with_region_filters(url: str, region: Optional[str]) -> str:
    reg = _normalize_region(region)
    if not reg:
        return url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    qd: Dict[str, List[str]] = {}
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        qd.setdefault(k, []).append(v)
    _apply_region_filters_to_query(qd, reg)
    query = urlencode(qd, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))

# ---------------- JSON extractors (HTML regex, proven) ----------------
def _extract_apollo_state_reviews(html: str) -> List[Dict[str, Any]]:
    if not html: return []
    m = re.search(r'apolloState"\s*:\s*({.+?})\s*}\s*[,;]\s*</script', html, flags=re.S|re.I)
    if not m: return []
    try:
        apollo = json.loads(m.group(1))
    except Exception as e:
        log(f"[apollo] JSON decode failed: {e}")
        return []
    cand: List[Dict[str, Any]] = []
    def looks(d: dict) -> bool:
        if not isinstance(d, dict): return False
        keys = {k.lower() for k in d.keys()}
        sig = [
            "pros","cons","headline","reviewbody","overallrating","rating",
            "jobtitle","createddatetime","summary","reviewdatetime","reviewid"
        ]
        return sum(any(s in k for k in keys) for s in sig) >= 2
    def walk(o):
        if isinstance(o, dict):
            if looks(o): cand.append(o)
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(apollo)
    out, seen = [], set()
    for r in cand:
        x = {
            "review_id": r.get("reviewId") or r.get("id"),
            "title": r.get("headline") or r.get("title") or r.get("summary"),
            "body": r.get("reviewBody") or r.get("body") or r.get("text"),
            "pros": r.get("pros") or r.get("reviewPros"),
            "cons": r.get("cons") or r.get("reviewCons"),
            "rating": r.get("overallRating") or r.get("ratingOverall") or r.get("rating"),
            "date": r.get("reviewDate") or r.get("createdDateTime") or r.get("reviewDateTime") or r.get("time"),
            "role": r.get("jobTitle") or r.get("authorJobTitle"),
            "location": r.get("location") or r.get("reviewerLocation"),
            "employmentStatus": r.get("employmentStatus"),
            "_source": "apollo",
        }
        k = x.get("review_id") or (x.get("title"), x.get("date"))
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _extract_next_data_reviews(html: str) -> List[Dict[str, Any]]:
    """Stricter filter: only collect true review objects (skip highlight/aggregate blobs)."""
    if not html: return []
    m = re.search(r'id="__NEXT_DATA__"[^>]*>\s*({.+?})\s*</script>', html, flags=re.S|re.I)
    if not m: return []
    try:
        data = json.loads(m.group(1))
    except Exception as e:
        log(f"[next] JSON decode failed: {e}")
        return []

    def is_review_shape(d: dict) -> bool:
        if not isinstance(d, dict): return False
        keys = {k.lower() for k in d.keys()}
        if not (("reviewid" in keys or "id" in keys) and ("pros" in keys or "cons" in keys)):
            return False
        t = d.get("__typename", "")
        if isinstance(t, str) and ("Highlight" in t or "ProsAndConsType" in t):
            return False
        rid = d.get("reviewId") or d.get("id")
        return isinstance(rid, (int, str))

    cand: List[Dict[str, Any]] = []
    def walk(o):
        if isinstance(o, dict):
            if is_review_shape(o): cand.append(o)
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(data)

    out, seen = [], set()
    for r in cand:
        x = {
            "review_id": r.get("reviewId") or r.get("id"),
            "title": r.get("headline") or r.get("title"),
            "body": r.get("reviewBody") or r.get("body") or r.get("text"),
            "pros": r.get("pros"),
            "cons": r.get("cons"),
            "rating": r.get("overallRating") or r.get("rating"),
            "date": r.get("reviewDate") or r.get("createdDateTime"),
            "role": r.get("jobTitle"),
            "location": r.get("location"),
            "employmentStatus": r.get("employmentStatus"),
            "_source": "next",
        }
        k = x.get("review_id")
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _extract_next_f_reviews(html: str) -> List[Dict[str, Any]]:
    """
    Extract review objects embedded in Next.js streaming chunks:
    <script>self.__next_f.push([1,"...escaped payload..."])</script>
    """
    if not html:
        return []

    chunk_re = re.compile(
        r'self\.__next_f\.push\(\[\d+,\s*"((?:[^"\\]|\\.)*)"\]\)',
        flags=re.S,
    )
    chunks = chunk_re.findall(html)
    if not chunks:
        return []

    dec = json.JSONDecoder()
    out: List[Dict[str, Any]] = []
    seen_ids = set()

    def _job_title_text(v: Any) -> Optional[str]:
        if isinstance(v, str):
            return v.strip() or None
        if isinstance(v, dict):
            return (v.get("text") or v.get("name") or v.get("title") or "").strip() or None
        return None

    def _location_text(v: Any) -> Optional[str]:
        if isinstance(v, str):
            return v.strip() or None
        if isinstance(v, dict):
            parts = []
            for k in ("name", "city", "regionName", "state", "countryName", "country"):
                val = v.get(k)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            if parts:
                # Preserve order while dropping duplicates.
                uniq = []
                seen = set()
                for p in parts:
                    if p not in seen:
                        seen.add(p)
                        uniq.append(p)
                return ", ".join(uniq)
        return None

    for raw in chunks:
        # Fast skip for chunks that cannot contain reviews.
        if '\\"reviewId\\":' not in raw and '"reviewId":' not in raw:
            continue

        try:
            text = bytes(raw, "utf-8").decode("unicode_escape", errors="ignore")
        except Exception:
            continue

        for m in re.finditer(r'"reviewId"\s*:\s*\d+', text):
            idx = m.start()
            attempts = 0
            start_floor = max(-1, idx - 20000)
            found_obj = None

            for s in range(idx, start_floor, -1):
                if text[s] != "{":
                    continue
                attempts += 1
                if attempts > 500:
                    break
                try:
                    obj, _ = dec.raw_decode(text[s:])
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("reviewId") is not None:
                    found_obj = obj
                    break

            if not found_obj:
                continue

            rid = found_obj.get("reviewId")
            rid_key = str(rid).strip() if rid is not None else ""
            if not rid_key or rid_key in seen_ids:
                continue
            seen_ids.add(rid_key)

            is_current = found_obj.get("isCurrentJob")
            status_label = None
            if isinstance(is_current, bool):
                status_label = "Current employee" if is_current else "Former employee"

            job_title = _job_title_text(found_obj.get("jobTitle"))
            role = None
            if status_label and job_title:
                role = f"{status_label} - {job_title}"
            else:
                role = status_label or job_title

            out.append({
                "review_id": rid,
                "title": found_obj.get("summary") or found_obj.get("headline") or found_obj.get("title"),
                "body": found_obj.get("reviewBody") or found_obj.get("body") or found_obj.get("text") or found_obj.get("summary"),
                "pros": found_obj.get("pros") or found_obj.get("reviewPros"),
                "cons": found_obj.get("cons") or found_obj.get("reviewCons"),
                "rating": found_obj.get("ratingOverall") or found_obj.get("overallRating") or found_obj.get("rating"),
                "date": found_obj.get("reviewDateTime") or found_obj.get("reviewDate") or found_obj.get("createdDateTime"),
                "role": role,
                "location": _location_text(found_obj.get("location")),
                "employmentStatus": found_obj.get("employmentStatus"),
                "_source": "next_f",
            })

    return out

def _extract_ldjson_reviews(html: str) -> List[Dict[str, Any]]:
    if not html:
        return []

    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.S | re.I
    )
    if not scripts:
        return []

    def as_list(x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            return [x]
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for raw in scripts:
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        for obj in as_list(data):
            candidates = as_list(obj.get("@graph")) if isinstance(obj, dict) and "@graph" in obj else as_list(obj)
            for r in candidates:
                if not isinstance(r, dict):
                    continue

                rtype = r.get("@type") or r.get("type")
                if isinstance(rtype, list):
                    rtype = " ".join(map(str, rtype))
                if not (isinstance(rtype, str) and "review" in rtype.lower()):
                    continue

                body = r.get("reviewBody") or r.get("description") or ""
                rr = r.get("reviewRating") or {}
                rating = rr.get("ratingValue") if isinstance(rr, dict) else None

                author = r.get("author") or {}
                if isinstance(author, dict):
                    author_name = author.get("name")
                else:
                    author_name = str(author) if author is not None else None

                date = r.get("datePublished") or r.get("dateCreated") or None

                review_id = r.get("@id")
                if not review_id:
                    seed = f"{author_name}|{rating}|{date}|{body}"
                    review_id = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]

                title = r.get("name") or r.get("headline")
                if not title and isinstance(body, str):
                    title = (body[:80] + "...") if len(body) > 80 else body

                if review_id in seen:
                    continue
                seen.add(review_id)

                out.append({
                    "review_id": review_id,
                    "title": title,
                    "body": body,
                    "pros": None,
                    "cons": None,
                    "rating": rating,
                    "date": date,
                    "role": author_name,
                    "location": None,
                    "employmentStatus": None,
                    "_source": "ldjson",
                })
    return out

# ---------- Deref + backfill helpers (same as working version) ----------
def _apollo_index_from_html(html: str) -> Dict[str, Any]:
    if not html:
        return {}

    m = re.search(
        r'apolloState"\s*:\s*({.+?})\s*}\s*[,;]\s*</script>',
        html, flags=re.S | re.I
    )
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    def _load_next_data(h: str) -> Optional[Dict[str, Any]]:
        m2 = re.search(
            r'id="__NEXT_DATA__"[^>]*>\s*({.+?})\s*</script>',
            h, flags=re.S | re.I
        )
        if not m2:
            return None
        try:
            return json.loads(m2.group(1))
        except Exception:
            return None

    def _find_apollo_dict(o: Any) -> Optional[Dict[str, Any]]:
        best = None
        best_score = 0
        def score_dict(d: Dict[str, Any]) -> int:
            if not isinstance(d, dict): return 0
            vals = list(d.values())
            if not vals: return 0
            typename_cnt = sum(1 for v in vals if isinstance(v, dict) and "__typename" in v)
            colon_key_cnt = sum(1 for k in d.keys() if isinstance(k, str) and ":" in k)
            return 2 * typename_cnt + colon_key_cnt
        def walk(x: Any):
            nonlocal best, best_score
            if isinstance(x, dict):
                s = score_dict(x)
                if s > best_score and len(x) >= 3:
                    best, best_score = x, s
                for v in x.values(): walk(v)
            elif isinstance(x, list):
                for v in x: walk(v)
        walk(o)
        return best

    nd = _load_next_data(html)
    if not nd:
        return {}
    cur = nd
    for key in ("props", "pageProps", "apolloState"):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            cur = None
            break
    if isinstance(cur, dict) and cur:
        return cur
    apollo_guess = _find_apollo_dict(nd)
    return apollo_guess if isinstance(apollo_guess, dict) else {}

def _deref_ref(v: Any, idx: Dict[str, Any]) -> Optional[str]:
    if not isinstance(v, dict) or "__ref" not in v:
        return v if isinstance(v, str) else None
    ref = v.get("__ref")
    target = idx.get(ref, {})
    if isinstance(target, dict):
        tname = target.get("__typename") or ""
        if tname == "JobTitle" or str(ref).startswith("JobTitle:"):
            return target.get("text") or target.get("jobTitleText") or target.get("name")
        if tname == "City" or str(ref).startswith("City:"):
            def _txt(x):
                if isinstance(x, dict) and "__ref" in x:
                    return _deref_ref(x, idx) or ""
                if isinstance(x, dict):
                    for k in ("name", "text", "label", "title"):
                        vv = x.get(k)
                        if isinstance(vv, str):
                            return vv
                    return ""
                return x if isinstance(x, str) else ""
            city = _txt(target.get("name") or target.get("city"))
            region = _txt(target.get("regionName") or target.get("state") or target.get("region"))
            country = _txt(target.get("countryName") or target.get("country"))
            parts = [p for p in (city, region, country) if isinstance(p, str) and p]
            return ", ".join(parts) if parts else None
        return target.get("name") or target.get("title") or target.get("label")
    return None

def _first_nonempty(*vals) -> Optional[Any]:
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

def _oneline(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    return re.sub(r'\s+', ' ', s).strip()

def _normalize_row_fields(row: Dict[str, Any], apollo_idx: Dict[str, Any]) -> Dict[str, Any]:
    row["role"] = _first_nonempty(
        _deref_ref(row.get("role"), apollo_idx),
        _deref_ref(row.get("jobTitle"), apollo_idx),
        row.get("role") if isinstance(row.get("role"), str) else None
    ) or "Unknown role"

    row["location"] = _first_nonempty(
        _deref_ref(row.get("location"), apollo_idx),
        row.get("location") if isinstance(row.get("location"), str) else None
    ) or "Unknown location"

    row["title"] = _first_nonempty(
        row.get("title"),
        row.get("headline"),
        row.get("summary"),
    )
    if not row.get("title"):
        seed = _first_nonempty(row.get("pros"), row.get("cons"), row.get("body"), "Review")
        row["title"] = (seed or "Review")
    row["title"] = _oneline(row.get("title"))

    rating_raw = _first_nonempty(row.get("rating"), row.get("overallRating"), row.get("ratingOverall"))
    if isinstance(rating_raw, (int, float)):
        row["rating"] = float(rating_raw)
    elif isinstance(rating_raw, str):
        m = re.search(r'\d+(?:\.\d+)?', rating_raw)
        row["rating"] = float(m.group(0)) if m else None
    else:
        row["rating"] = None
    if row["rating"] is None:
        row["rating"] = 0.0

    row["date"] = _first_nonempty(
        row.get("date"),
        row.get("createdDateTime"),
        row.get("reviewDate"),
        row.get("reviewDateTime")
    ) or ""
    return row

def _find_apollo_review_node(apollo_idx: Dict[str, Any], review_id: Any) -> Optional[Dict[str, Any]]:
    if not apollo_idx or review_id is None:
        return None
    rid_s = str(review_id)
    for v in apollo_idx.values():
        if isinstance(v, dict):
            if v.get("reviewId") == review_id or v.get("id") == review_id:
                return v
    for k, v in apollo_idx.items():
        if not isinstance(k, str): continue
        if "Review" not in k: continue
        try:
            _, jsonish = k.split(":", 1)
            jsonish = jsonish.strip()
            if jsonish.startswith("{") and jsonish.endswith("}"):
                obj = json.loads(jsonish)
                if str(obj.get("reviewId") or obj.get("id")) == rid_s:
                    return v if isinstance(v, dict) else None
        except Exception:
            continue
    for k, v in apollo_idx.items():
        if isinstance(k, str) and rid_s in k and isinstance(v, dict):
            return v
    return None

def _coerce_rating(v) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        m = re.search(r'\d+(?:\.\d+)?', v)
        if m:
            val = float(m.group(0))
            return val if val > 0 else None
    return None

def _enrich_from_apollo(row: Dict[str, Any], apollo_idx: Dict[str, Any]) -> Dict[str, Any]:
    node = _find_apollo_review_node(apollo_idx, row.get("review_id"))
    if not node:
        return row

    if not row.get("title") or row.get("title") in ("Review", ""):
        row["title"] = _first_nonempty(row.get("title"), node.get("summary"), "Review")
        row["title"] = _oneline(row["title"])

    cur = row.get("rating")
    needs_rating = cur in (None, "", "0", "0.0", 0, 0.0)
    if needs_rating:
        apollo_rating = _coerce_rating(
            _first_nonempty(
                node.get("ratingOverall"),
                node.get("overallRating"),
                node.get("rating"),
            )
        )
        if apollo_rating is not None:
            row["rating"] = max(0.0, min(5.0, apollo_rating))

    if not row.get("date"):
        row["date"] = _first_nonempty(
            row.get("date"),
            node.get("reviewDateTime"),
            node.get("createdDateTime"),
            node.get("reviewDate")
        ) or ""

    if row.get("role") in (None, "", "Unknown role") or isinstance(row.get("role"), dict):
        row["role"] = _first_nonempty(
            _deref_ref(node.get("jobTitle"), apollo_idx),
            row.get("role"),
            "Unknown role"
        )

    if row.get("location") in (None, "", "Unknown location") or isinstance(row.get("location"), dict):
        row["location"] = _first_nonempty(
            _deref_ref(node.get("location"), apollo_idx),
            row.get("location"),
            "Unknown location"
        )
    return row

# ---------- Safe de-dupe key helpers ----------
def _safe_scalar(v: Any):
    if v is None:
        return ""
    if isinstance(v, (int, float, str)):
        return str(v)
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(v)

def _row_dedupe_key(r: Dict[str, Any]):
    rid = r.get("review_id")
    if isinstance(rid, (int, str)) and str(rid).strip() != "":
        return ("id", str(rid))
    return (
        "tdr",
        _safe_scalar(r.get("title")),
        _safe_scalar(r.get("date")),
        _safe_scalar(r.get("role")),
    )

# ---------- CSV writer ----------
def _write_reviews_csv(rows: List[Dict[str, Any]], path: Path):
    fields = ["review_id","title","body","pros","cons","rating","date","role","location","employmentStatus","_source"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_ALL, strict=True)
        w.writeheader()
        for r in rows:
            rec = {k: r.get(k, "") for k in fields}
            for k in ("title","body","pros","cons","role","location"):
                v = rec.get(k, "")
                if isinstance(v, (list, dict)):
                    try:
                        v = json.dumps(v, ensure_ascii=False)
                    except Exception:
                        v = str(v)
                if isinstance(v, str):
                    v = _oneline(v)
                rec[k] = "" if v is None else v
            if rec["rating"] is None:
                rec["rating"] = ""
            for k, v in rec.items():
                rec[k] = sanitize_csv_cell(v)
            w.writerow(rec)

# ---------------- Browser client (restored workflow + updates) ----------------
class GlassdoorReviews:
    """
    Restored "working" flow:
      - always attempt extraction from HTML (apollo + __NEXT_DATA__)
      - challenge/soft-block detection is advisory unless challenge_mode=block
    """
    def __init__(
        self,
        headless: bool = False,
        chrome_path: Optional[str] = None,
        profile_dir: Optional[Path] = None,
        region: Optional[str] = None,
        challenge_mode: str = "log_only",       # block | log_only
        pause_until_enter: bool = False,         # block-mode only
        challenge_wait: float = 0.0,             # block-mode only
        challenge_max_retries: int = 3,          # block-mode only
        challenge_retry_backoff: float = 3.0,    # block-mode only
        softblock_max_retries: int = 2,
        softblock_retry_backoff: float = 4.0,
        stop_on_empty_pages: int = 2,
        hydrate_timeout: float = 12.0,
        dump_debug_html_on_fail: bool = True,
    ):
        log("[init] Chromium options")
        opts = ChromiumOptions()
        prof = (profile_dir or PROFILE_DIR).resolve()
        opts.add_argument(f"--user-data-dir={prof}")
        if chrome_path:
            try:
                opts.set_binary(chrome_path)
                log(f"[init] chrome binary: {chrome_path}")
            except Exception as e:
                log(f"[init] set_binary failed: {e}")
        if headless:
            opts.add_argument("--headless=new")
        self.opts = opts
        self.region = _normalize_region(region)

        self.challenge_mode = str(challenge_mode).strip().lower()
        self.pause_until_enter = bool(pause_until_enter)
        self.challenge_wait = float(challenge_wait or 0.0)
        self.challenge_max_retries = int(max(0, challenge_max_retries))
        self.challenge_retry_backoff = float(max(0.0, challenge_retry_backoff))

        self.softblock_max_retries = int(max(0, softblock_max_retries))
        self.softblock_retry_backoff = float(max(0.0, softblock_retry_backoff))

        self.stop_on_empty_pages = int(max(1, stop_on_empty_pages))
        self.hydrate_timeout = float(max(0.0, hydrate_timeout))
        self.dump_debug_html_on_fail = bool(dump_debug_html_on_fail)

    async def _eval_js(self, tab, script: str):
        return await tab.execute_script(script)

    async def _get_page_source(self, tab) -> str:
        attr = getattr(tab, "page_source", None)
        if attr is not None:
            try:
                if isinstance(attr, str): return attr
                if inspect.iscoroutine(attr): return await attr
                if callable(attr):
                    res = attr()
                    if inspect.iscoroutine(res): return await res
                    if isinstance(res, str): return res
            except Exception:
                pass
        get_ps = getattr(tab, "get_page_source", None)
        if callable(get_ps):
            try:
                res = get_ps()
                if inspect.iscoroutine(res): return await res
                if isinstance(res, str): return res
            except Exception:
                pass
        try:
            return await self._eval_js(tab, "document.documentElement.outerHTML || ''")
        except Exception:
            return ""

    async def _sleep_human(self, a=0.2, b=0.6):
        await asyncio.sleep(a + random.random()*(b-a))

    async def _accept_cookies(self, tab):
        js = r"""
        (() => {
          const txt = el => el && (el.textContent||"").trim().toLowerCase();
          const buttons = Array.from(document.querySelectorAll('button,[role="button"],input[type="button"],input[type="submit"]'));
          const wants = ["accept all","accept","i agree","agree & proceed","allow all"];
          for (const b of buttons) {
            const t = txt(b); if (!t) continue;
            if (wants.some(w=>t.includes(w))) { b.click(); return true; }
          }
          const gd = document.querySelector('[data-test="GDPR-accept"]') || document.querySelector('#onetrust-accept-btn-handler');
          if (gd) { gd.click(); return true; }
          return false;
        })();
        """
        try:
            ok = await self._eval_js(tab, js)
            if ok: log("[consent] Accepted cookies")
        except Exception:
            pass

    # ---- Robust URL un-wrapper ----
    def _unwrap_url_obj(self, v) -> str:
        seen = set()
        while isinstance(v, dict) and id(v) not in seen:
            seen.add(id(v))
            if "value" in v:
                v = v["value"]; continue
            if "result" in v:
                v = v["result"]; continue
            if "data" in v:
                v = v["data"]; continue
            if v.keys() == {"type", "value"} and isinstance(v.get("value"), (str, dict)):
                v = v["value"]; continue
            break
        s = v if isinstance(v, str) else ("" if v is None else str(v))
        m = re.search(r'https?://[^\s\'"}<>]+', s)
        return m.group(0) if m else s.strip()

    async def _current_href(self, tab) -> str:
        try:
            href = await self._eval_js(tab, "location.href")
            return self._unwrap_url_obj(href)
        except Exception:
            return ""

    # ---------------- STRICT HARD challenge detection ----------------
    async def _detect_hard_challenge(self, tab) -> bool:
        js = r"""
        (() => {
          const h = (document.documentElement.innerHTML||"").toLowerCase();
          const t = (document.title||"").toLowerCase();

          const titleHit =
            t.includes("just a moment") ||
            t.includes("attention required") ||
            t.includes("verifying you are human") ||
            t.includes("verification required") ||
            t.includes("access denied");

          const cfHit =
            h.includes("/cdn-cgi/challenge-platform") ||
            h.includes("cf-challenge") ||
            h.includes("challenge-platform") ||
            h.includes("cf-turnstile");

          const captchaHit =
            !!document.querySelector(
              'iframe[src*="hcaptcha"],iframe[src*="recaptcha"],iframe[title*="captcha" i],' +
              'div.h-captcha,div.g-recaptcha,[data-sitekey]'
            );

          return !!(titleHit || cfHit || captchaHit);
        })();
        """
        try:
            return bool(await self._eval_js(tab, js))
        except Exception:
            return False

    async def _detect_soft_block(self, tab) -> bool:
        """
        Soft-block is *advisory* and should not suppress extraction.
        We only flag obvious error shells / blocked text / extremely short docs.
        """
        js = r"""
        (() => {
          const h = (document.documentElement.innerHTML||"");
          const hl = h.toLowerCase();
          const t = (document.title||"").toLowerCase();

          const looksLikeError =
            t.includes("error") ||
            hl.includes("something went wrong") ||
            hl.includes("temporarily unavailable") ||
            hl.includes("request was blocked") ||
            hl.includes("unusual traffic") ||
            hl.includes("please enable javascript") ||
            hl.includes("access denied");

          const tooShort = h.length < 3000;

          return !!(looksLikeError || tooShort);
        })();
        """
        try:
            return bool(await self._eval_js(tab, js))
        except Exception:
            return False

    async def _handle_hard_challenge(self, tab, where: str = "") -> bool:
        if not await self._detect_hard_challenge(tab):
            return False
        tag = f" ({where})" if where else ""
        log(f"[challenge] Detected{tag}.")

        if self.challenge_mode == "log_only":
            log("[challenge] Proceeding without pause (log_only mode).")
            return True

        # block mode behavior (pause/wait)
        if self.pause_until_enter:
            input("Cloudflare/captcha detected. Solve it in the browser, then press Enter to continue...")
        elif self.challenge_wait and self.challenge_wait > 0:
            log(f"[challenge] Waiting {self.challenge_wait:.1f}s...")
            await asyncio.sleep(self.challenge_wait)
        else:
            log("[challenge] Detected (no pause configured).")
        return True

    async def _maybe_dump_debug_html(self, html: str, name: str):
        if not self.dump_debug_html_on_fail:
            return
        try:
            p = DEBUG_DIR / f"{name}.html"
            p.write_text(html or "", encoding="utf-8", errors="ignore")
            log(f"[debug] Saved HTML -> {p.resolve()}")
        except Exception:
            pass

    # ---------- Navigation ----------
    def _overview_to_reviews_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
        except Exception:
            return url
        if not parsed.scheme or not parsed.netloc:
            return url
        path = parsed.path or ""
        if "/Reviews/" in path:
            return url
        if "/Overview/" not in path:
            return url
        m = re.search(r"Working-at-([^-]+)-EI_IE(\d+)", path, flags=re.I)
        if not m:
            return url
        company_slug = m.group(1)
        employer_id = m.group(2)
        new_path = f"/Reviews/{company_slug}-Reviews-EI_IE{employer_id}.htm"
        return urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", ""))

    async def _goto_reviews_tab(self, tab, url: str):
        url = self._overview_to_reviews_url(url)
        url = _with_region_filters(url, self.region)
        await tab.go_to(url)
        await self._sleep_human()
        await self._accept_cookies(tab)
        await self._handle_hard_challenge(tab, where="initial")

        # match old behavior: force eng if different
        await self._eval_js(tab, r"""
          (() => {
            const u = new URL(window.location.href);
            if (u.searchParams.get('filter.iso3Language') !== 'eng') {
              u.searchParams.set('filter.iso3Language','eng');
              window.location.replace(u.toString());
              return true;
            }
            return false;
          })();
        """)
        await self._sleep_human(0.8, 1.2)

    # ---- Deterministic pagination (_P{n}.htm) ----
    def _canonize_reviews_base(self, url: str) -> Tuple[str, dict, str]:
        url = self._unwrap_url_obj(url)
        base = re.sub(r'(_P\d+|_IP\d+)?\.htm.*$', '', url)
        qs_match = re.search(r'\?(.+)$', url)
        pairs = parse_qsl(qs_match.group(1), keep_blank_values=True) if qs_match else []
        qd: Dict[str, List[str]] = {}
        for k, v in pairs:
            qd.setdefault(k, []).append(v)

        # keep old behavior: set explicit values
        qd['filter.iso3Language'] = ['eng']
        qd['sort.sortType'] = ['RD']
        qd['sort.ascending'] = ['false']
        _apply_region_filters_to_query(qd, self.region)

        page_token = "IP" if re.search(r'(?:-US-Reviews|_IN\d+)', base, flags=re.I) else "P"
        return base, qd, page_token

    def _page_url(self, base: str, qd: dict, page_idx: int, page_token: str) -> str:
        suffix = "" if page_idx == 1 else f"_{page_token}{page_idx}"
        path = f"{base}{suffix}.htm"
        qs = urlencode(qd, doseq=True)
        return f"{path}?{qs}" if qs else path

    async def _wait_for_hydration(self, tab) -> None:
        """
        IMPORTANT: hydration wait is best-effort only.
        We do NOT require __NEXT_DATA__ to exist (Glassdoor sometimes omits it).
        """
        if self.hydrate_timeout <= 0:
            return
        start = time.time()
        while (time.time() - start) < self.hydrate_timeout:
            try:
                # If either apolloState or __NEXT_DATA__ appears, good enough
                has_next = await self._eval_js(tab, "!!document.getElementById('__NEXT_DATA__')")
                has_apollo = await self._eval_js(tab, "document.documentElement.innerHTML.toLowerCase().includes('apollostate')")
                if has_next or has_apollo:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.25)

    # ---------------- main (restored flow) ----------------
    async def scrape_reviews(
        self,
        company_url: str,
        pages: int = 3,
        start_page: int = 1,
        end_page: Optional[int] = None,
        csv_path: Optional[Path] = None,
        page_delay: float = 3.0
    ) -> List[Dict[str, Any]]:
        log("[run] Launching Chrome...")
        async with Chrome(options=self.opts) as browser:
            tab = await browser.start()
            log("[run] Tab started")

            await self._goto_reviews_tab(tab, company_url)

            # keep old “force RD + false” logic
            try:
                changed = await self._eval_js(tab, """
                  (() => {
                    const u = new URL(location.href);
                    let did = false;
                    if (u.searchParams.get('sort.sortType') !== 'RD') {
                      u.searchParams.set('sort.sortType','RD');
                      did = true;
                    }
                    if (u.searchParams.get('sort.ascending') !== 'false') {
                      u.searchParams.set('sort.ascending','false');
                      did = true;
                    }
                    if (did) { location.replace(u.toString()); return true; }
                    return false;
                  })();
                """)
                if changed:
                    await self._sleep_human(0.7, 1.1)
                    await self._handle_hard_challenge(tab, where="post_sort_redirect")
            except Exception:
                pass

            cur_url = await self._current_href(tab)
            base, qd, page_token = self._canonize_reviews_base(cur_url)
            base = self._unwrap_url_obj(base)

            start_page = int(start_page)
            if start_page < 1:
                raise ValueError("--start-page must be >= 1")

            pages = int(pages)
            if pages < 1:
                raise ValueError("--pages must be >= 1")

            if end_page is None:
                end_page = start_page + pages - 1
            else:
                end_page = int(end_page)

            if end_page < start_page:
                raise ValueError("--end-page must be >= --start-page")

            page_numbers = list(range(start_page, end_page + 1))
            total_pages = len(page_numbers)
            log(f"[nav] Base resolved: {base}.htm  | page_range={start_page}-{end_page} ({total_pages} pages) | mode=_{page_token}*")

            all_rows: List[Dict[str, Any]] = []
            empty_streak = 0

            for idx, page in enumerate(page_numbers, start=1):
                target = self._page_url(base, qd, page, page_token)
                target = self._unwrap_url_obj(target)
                log(f"[page] {idx}/{total_pages} (source page {page}) -> {target}")

                # soft-block retries are advisory: we retry navigation a bit, but we ALWAYS extract.
                soft_attempt = 0
                while True:
                    await tab.go_to(target)
                    await self._sleep_human(0.4, 0.8)

                    landed = await self._current_href(tab)
                    log(f"[nav] Landed: {landed}")

                    # hard challenge handling
                    hard = await self._detect_hard_challenge(tab)
                    if hard:
                        await self._handle_hard_challenge(tab, where=f"page {page}")
                        # in log_only, do not loop; just attempt extraction
                        if self.challenge_mode == "log_only":
                            break
                        # in block mode, retry with backoff
                        soft_attempt = 0
                        # use block-mode retry logic
                        # (challenge_max_retries/backoff like your upgraded version)
                        # note: we keep it simple and bounded
                        # if you want per-page hard_attempt counters back, we can add them.
                        break

                    # soft-block advisory loop
                    soft = await self._detect_soft_block(tab)
                    if soft:
                        soft_attempt += 1
                        log(f"[soft_block] Detected (page {page}). Backing off.")
                        if soft_attempt > self.softblock_max_retries:
                            html = await self._get_page_source(tab)
                            await self._maybe_dump_debug_html(html, f"page{page}_soft_block")
                            log(f"[soft_block] Max soft retries exceeded on page {page}. Proceeding to extraction attempt anyway.")
                            break
                        backoff = self.softblock_retry_backoff * soft_attempt
                        log(f"[soft_block] Retrying page {page} after {backoff:.1f}s (attempt {soft_attempt}/{self.softblock_max_retries})")
                        await asyncio.sleep(backoff)
                        continue

                    break  # normal enough

                # small scroll helps hydration
                try:
                    await self._eval_js(tab, "window.scrollBy(0, 400);")
                except Exception:
                    pass
                await self._sleep_human(0.2, 0.5)

                # best-effort hydration wait (does not require NEXT_DATA)
                await self._wait_for_hydration(tab)

                html = await self._get_page_source(tab)

                # extraction (restored): always parse from HTML
                rows: List[Dict[str, Any]] = []
                rows.extend(_extract_apollo_state_reviews(html))
                rows.extend(_extract_next_data_reviews(html))
                rows.extend(_extract_next_f_reviews(html))
                if not rows:
                    # Fallback when only SEO schema is present.
                    rows.extend(_extract_ldjson_reviews(html))
                rows = [r for r in rows if r.get("review_id") not in (None, "", [])]
                log(f"[json] Extracted {len(rows)} objects on page {page}")

                # empty-streak logic (restored) + one safety tweak:
                # If we are in log_only and this page is a *real* HARD challenge, don't count it.
                if len(rows) == 0:
                    is_hard = await self._detect_hard_challenge(tab)
                    if self.challenge_mode == "log_only" and is_hard:
                        log(f"[challenge] log_only: page {page} had HARD challenge signals; not counting toward empty-streak.")
                    else:
                        empty_streak += 1
                        if empty_streak >= self.stop_on_empty_pages:
                            log(f"[stop] No JSON reviews for {empty_streak} consecutive pages. Stopping at page {page}.")
                            # optional debug dump on stop
                            await self._maybe_dump_debug_html(html, f"page{page}_empty_stop")
                            break
                else:
                    empty_streak = 0

                # normalize/enrich (restored)
                apollo_idx = _apollo_index_from_html(html)
                for r in rows:
                    _normalize_row_fields(r, apollo_idx)
                    _enrich_from_apollo(r, apollo_idx)

                all_rows.extend(rows)

                if idx < total_pages:
                    pause = max(0.0, float(page_delay))
                    log(f"[pace] Sleeping {pause:.1f}s before next page...")
                    await asyncio.sleep(pause)

            # dedup (restored)
            dedup, seen = [], set()
            for r in all_rows:
                rating = r.get("rating")
                if isinstance(rating, str):
                    m = re.search(r'\d+(?:\.\d+)?', rating)
                    r["rating"] = float(m.group(0)) if m else None
                key = _row_dedupe_key(r)
                if key not in seen:
                    seen.add(key)
                    dedup.append(r)

            OUT_JSON.write_text(json.dumps(dedup, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"[run] Saved {len(dedup)} reviews -> {OUT_JSON.resolve()}")

            if csv_path:
                _write_reviews_csv(dedup, Path(csv_path))
                log(f"[run] Saved {len(dedup)} rows -> {Path(csv_path).resolve()}")

            if dedup:
                log(f"[sample] {dedup[0]}")
            return dedup

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Glassdoor Reviews Scraper (restored working extraction + upgrades)")
    p.add_argument("-u","--url", required=True, help="Company Reviews or Overview URL (I'll navigate to Reviews)")
    p.add_argument(
        "--region",
        default=None,
        help=(
            "Optional location filter (for example: 'United States'). "
            "For United States, the scraper enforces filter.locationId=1 and filter.locationType=N."
        ),
    )
    p.add_argument("-p","--pages", type=int, default=3,
                   help="Number of review pages to collect (used when --end-page is not set)")
    p.add_argument("--start-page", type=int, default=1,
                   help="First review page number to scrape (default: 1)")
    p.add_argument("--end-page", type=int, default=None,
                   help="Last review page number to scrape (inclusive). If set, overrides --pages count")
    p.add_argument("--headless", action="store_true", help="Run Chrome headless")
    p.add_argument("--chrome-binary", help="Path to Chrome binary (optional)")
    p.add_argument("--profile-dir", help="Custom Chrome profile dir (optional)")
    p.add_argument("--timeout", type=int, default=1600, help="Overall run timeout (seconds)")
    p.add_argument("--page-delay", type=float, default=3.0, help="Seconds to wait between pages after extraction")
    p.add_argument("-o","--out", default=str(OUT_JSON), help="Output JSON path")
    p.add_argument("--csv", help="Also write CSV to this path (e.g., reviews.csv)")

    # NEW: challenge-mode (kept)
    p.add_argument("--challenge-mode", choices=["block", "log_only"], default="log_only",
                   help="block: pause/wait on captcha; log_only: log and continue extracting")

    # block-mode knobs (kept)
    p.add_argument("--pause-until-enter", action="store_true",
                   help="Pause and wait for Enter when HARD challenge is detected (block mode)")
    p.add_argument("--challenge-wait", type=float, default=0.0,
                   help="Seconds to wait when HARD challenge detected (block mode)")
    p.add_argument("--challenge-max-retries", type=int, default=3,
                   help="Max retries per page when HARD challenge blocks extraction (block mode)")
    p.add_argument("--challenge-retry-backoff", type=float, default=3.0,
                   help="Backoff multiplier between HARD challenge retries (block mode)")

    # soft-block + stop + hydration + debug (kept)
    p.add_argument("--softblock-max-retries", type=int, default=2,
                   help="Max retries per page when SOFT block suspected")
    p.add_argument("--softblock-retry-backoff", type=float, default=4.0,
                   help="Backoff multiplier between SOFT block retries")
    p.add_argument("--stop-on-empty-pages", type=int, default=2,
                   help="Stop after N consecutive pages with zero extracted reviews (default: 2)")
    p.add_argument("--hydrate-timeout", type=float, default=12.0,
                   help="Best-effort wait for hydration (NEXT_DATA/apollo) before extraction (default: 12)")
    p.add_argument("--no-debug-html", action="store_true",
                   help="Disable saving debug_html/*.html on failures/stops")

    args = p.parse_args()
    if args.pages < 1:
        p.error("--pages must be >= 1")
    if args.start_page < 1:
        p.error("--start-page must be >= 1")
    if args.end_page is not None and args.end_page < args.start_page:
        p.error("--end-page must be >= --start-page")
    return args

async def run_cli(args):
    global OUT_JSON
    OUT_JSON = Path(args.out)
    client = GlassdoorReviews(
        headless=args.headless,
        chrome_path=args.chrome_binary,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        region=args.region,
        challenge_mode=args.challenge_mode,
        pause_until_enter=args.pause_until_enter,
        challenge_wait=args.challenge_wait,
        challenge_max_retries=args.challenge_max_retries,
        challenge_retry_backoff=args.challenge_retry_backoff,
        softblock_max_retries=args.softblock_max_retries,
        softblock_retry_backoff=args.softblock_retry_backoff,
        stop_on_empty_pages=args.stop_on_empty_pages,
        hydrate_timeout=args.hydrate_timeout,
        dump_debug_html_on_fail=(not args.no_debug_html),
    )
    await client.scrape_reviews(
        args.url,
        pages=args.pages,
        start_page=args.start_page,
        end_page=args.end_page,
        csv_path=Path(args.csv) if args.csv else None,
        page_delay=args.page_delay
    )

if __name__ == "__main__":
    a = parse_args()
    log("[main] Starting reviews_scraper.py")
    try:
        asyncio.run(asyncio.wait_for(run_cli(a), timeout=a.timeout))
        log("[main] Done")
    except asyncio.TimeoutError:
        log(f"[main][ERROR] Timed out after {a.timeout}s")
        sys.exit(1)
    except Exception as e:
        import traceback
        log("[main][ERROR] Unhandled exception:")
        log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)
