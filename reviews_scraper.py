# reviews_scraper.py
import asyncio, json, re, inspect, sys, random, argparse, time, csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import parse_qsl, urlencode

from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions

PROFILE_DIR = Path("chrome-profile")
OUT_JSON    = Path("reviews.json")
DEBUG_DIR   = Path("debug_html")
DEBUG_DIR.mkdir(exist_ok=True)

def log(msg: str): print(msg, flush=True)

# ---------------- JSON extractors (embedded page caches) ----------------
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
            "jobtitle","createddatetime","summary","reviewdatetime"
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
    # Fix #1: robust __NEXT_DATA__ extraction
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

# ---------- Deref + backfill helpers ----------
def _apollo_index_from_html(html: str) -> Dict[str, Any]:
    """
    Return the Apollo normalized store as a dict mapping keys like
    'JobTitle:239483' → {...}. Works whether the store is embedded as a
    standalone <script> or nested inside __NEXT_DATA__.
    """
    if not html:
        return {}

    # Attempt A: standalone apolloState blob
    m = re.search(
        r'apolloState"\s*:\s*({.+?})\s*}\s*[,;]\s*</script>',
        html, flags=re.S | re.I
    )
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Attempt B: dig into __NEXT_DATA__
    def _load_next_data(h: str) -> Optional[Dict[str, Any]]:
        # Fix #1 (second occurrence)
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
    """Resolve {'__ref': 'Type:ID'} into a readable string; return None if can't."""
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
                        v = x.get(k)
                        if isinstance(v, str):
                            return v
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
    """Ensure title, rating, date, role, location are not null. Mutates & returns row."""
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
    """Locate the Apollo object for this review_id (keys vary: EmployerReview, EmployerReviewRG, etc.)."""
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
    """Parse a rating into float if > 0; else None."""
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        m = re.search(r'\d+(?:\.\d+)?', v)
        if m:
            val = float(m.group(0))
            return val if val > 0 else None
    return None

def _enrich_from_apollo(row: Dict[str, Any], apollo_idx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Apollo's review node to fill missing/placeholder fields:
    title ← summary, rating ← ratingOverall, date ← reviewDateTime,
    role/location ← deref(jobTitle/location).
    """
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
    """Return a hashable scalar usable in a set/dict key."""
    if v is None:
        return ""
    if isinstance(v, (int, float, str)):
        return str(v)
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(v)

def _row_dedupe_key(r: Dict[str, Any]):
    """Prefer review_id; else fall back to (title, date, role) after scalarizing."""
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
    """Write normalized/enriched rows to CSV."""
    fields = ["review_id","title","body","pros","cons","rating","date","role","location","employmentStatus","_source"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        # Fix #6: quote all fields for Excel-friendliness
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_ALL, strict=True)
        w.writeheader()
        for r in rows:
            rec = {k: r.get(k, "") for k in fields}
            # sanitize text-ish columns to one line
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
            w.writerow(rec)

# ---------------- Browser client (JSON-only) ----------------
class GlassdoorReviews:
    def __init__(self, headless: bool = False, chrome_path: Optional[str] = None, profile_dir: Optional[Path]=None):
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

    # --------- tab helpers ----------
    async def _eval_js(self, tab, script: str):
        return await tab.execute_script(script)

    async def _get_page_source(self, tab) -> str:
        attr = getattr(tab, "page_source", None)
        if attr is not None:
            if isinstance(attr, str): return attr
            if inspect.iscoroutine(attr): return await attr
            if callable(attr):
                res = attr()
                if inspect.iscoroutine(res): return await res
                if isinstance(res, str): return res
        get_ps = getattr(tab, "get_page_source", None)
        if callable(get_ps):
            res = get_ps()
            if inspect.iscoroutine(res): return await res
            if isinstance(res, str): return res
        try:
            return await self._eval_js(tab, "document.documentElement.outerHTML")
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

    async def _detect_challenge(self, tab) -> bool:
        js = r"""
        (() => {
          const h = document.documentElement.innerHTML.toLowerCase();
          const t = (document.title||"").toLowerCase();
          const cf = h.includes('cf-challenge') || t.includes('just a moment') || t.includes('verifying');
          const cap = h.includes('captcha') || h.includes('hcaptcha');
          return cf || cap;
        })();
        """
        try: return bool(await self._eval_js(tab, js))
        except Exception: return False

    async def _note_challenge(self, tab):
        try:
            if await self._detect_challenge(tab):
                log("[challenge] Detected (log-only). Proceeding without blocking.")
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

    async def _goto_reviews_tab(self, tab, url: str):
        await tab.go_to(url)
        await self._sleep_human()
        await self._accept_cookies(tab)
        await self._note_challenge(tab)

        try:
            is_reviews = await self._eval_js(tab, "(()=>/\\/reviews/i.test(location.pathname))();")
        except Exception:
            is_reviews = False

        if not is_reviews:
            import re as _re
            path = await self._eval_js(tab, "location.pathname") or ""
            if not _re.search(r'-US-Reviews', path, flags=_re.I):
                base = _re.sub(r'(?:_P\\d+|_IP\\d+)?\\.htm.*$', '', (await self._current_href(tab)))
                await tab.go_to(base + "/Reviews.htm")

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

    # ---- JSON helpers ----
    async def _extract_ids_from_html(self, html: str) -> set:
        ids = set()
        for r in (_extract_apollo_state_reviews(html) or []) + (_extract_next_data_reviews(html) or []):
            rid = r.get("review_id")
            if rid is not None:
                ids.add(rid)
        return ids

    # ---- Deterministic pagination (_P{n}.htm) ----
    def _canonize_reviews_base(self, url: str) -> (str, dict, str):
        url = self._unwrap_url_obj(url)
        base = re.sub(r'(_P\d+|_IP\d+)?\.htm.*$', '', url)
        qs_match = re.search(r'\?(.+)$', url)
        pairs = parse_qsl(qs_match.group(1), keep_blank_values=True) if qs_match else []
        qd: Dict[str, List[str]] = {}
        for k, v in pairs:
            qd.setdefault(k, []).append(v)
        qd['filter.iso3Language'] = ['eng']
        qd['sort.sortType'] = ['RD']
        qd['sort.ascending'] = ['false']
        page_token = "IP" if re.search(r'(?:-US-Reviews|_IN\d+)', base, flags=re.I) else "P"
        return base, qd, page_token

    def _page_url(self, base: str, qd: dict, page_idx: int, page_token: str) -> str:
        suffix = "" if page_idx == 1 else f"_{page_token}{page_idx}"
        path = f"{base}{suffix}.htm"
        qs = urlencode(qd, doseq=True) 
        return f"{path}?{qs}" if qs else path

    # ---------------- main (JSON-only) ----------------
    async def scrape_reviews(self, company_url: str, pages: int = 3, csv_path: Optional[Path] = None,
                             page_delay: float = 3.0) -> List[Dict[str, Any]]:
        log("[run] Launching Chrome…")
        async with Chrome(options=self.opts) as browser:
            tab = await browser.start(); log("[run] Tab started")

            await self._goto_reviews_tab(tab, company_url)

            # Keep sort=newest via JS (no waiter), then re-fetch URL
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
                if changed: await self._sleep_human(0.7, 1.1)
            except Exception:
                pass

            # Re-fetch URL after potential replace (but no explicit waiter)
            cur_url = await self._current_href(tab)
            base, qd, page_token = self._canonize_reviews_base(cur_url)
            base = self._unwrap_url_obj(base)
            log(f"[nav] Base resolved: {base}.htm  | pages={pages} | mode=_{page_token}*")

            all_rows: List[Dict[str, Any]] = []
            prev_ids: set = set()
            empty_streak = 0  # Fix #3: early stop on consecutive empty JSON pages

            for page in range(1, max(1, pages)+1):
                target = self._page_url(base, qd, page, page_token)
                target = self._unwrap_url_obj(target)
                log(f"[page] {page}/{pages} → {target}")
                await tab.go_to(target)
                await self._sleep_human(0.4, 0.8)
                await self._note_challenge(tab)

                landed = await self._current_href(tab)
                log(f"[nav] Landed: {landed}")

                try: await self._eval_js(tab, "window.scrollBy(0, 400);")
                except Exception: pass
                await self._sleep_human(0.2, 0.5)

                html = await self._get_page_source(tab)
                ids_now = await self._extract_ids_from_html(html)
                if page > 1 and ids_now and ids_now == prev_ids:
                    log("[warn] JSON IDs unchanged; page may not have advanced (rate-limit/CF). Continuing.")
                else:
                    prev_ids = ids_now

                apollo_idx = _apollo_index_from_html(html)

                rows: List[Dict[str, Any]] = []
                rows.extend(_extract_apollo_state_reviews(html))
                rows.extend(_extract_next_data_reviews(html))
                rows = [r for r in rows if r.get("review_id") not in (None, "", [])]

                log(f"[json] Extracted {len(rows)} objects on page {page}")

                # Fix #3: early stop condition
                if len(rows) == 0:
                    empty_streak += 1
                    if empty_streak >= 2:
                        log(f"[stop] No JSON reviews for {empty_streak} consecutive pages. Stopping at page {page}.")
                        break
                else:
                    empty_streak = 0

                for r in rows:
                    _normalize_row_fields(r, apollo_idx)
                    _enrich_from_apollo(r, apollo_idx)

                all_rows.extend(rows)

                # ---- NEW: configurable pause between pages
                if page < pages:
                    pause = max(0.0, float(page_delay))
                    log(f"[pace] Sleeping {pause:.1f}s before next page…")
                    await asyncio.sleep(pause)

            # de-dupe & normalize
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
            log(f"[run] Saved {len(dedup)} reviews → {OUT_JSON.resolve()}")

            if csv_path:
                _write_reviews_csv(dedup, Path(csv_path))
                log(f"[run] Saved {len(dedup)} rows → {Path(csv_path).resolve()}")

            if dedup: log(f"[sample] {dedup[0]}")
            return dedup

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Glassdoor Reviews Scraper (JSON-only, deterministic pagination)")
    p.add_argument("-u","--url", required=True, help="Company Reviews or Overview URL (I'll navigate to Reviews)")
    p.add_argument("-p","--pages", type=int, default=3, help="Number of review pages to collect")
    p.add_argument("--headless", action="store_true", help="Run Chrome headless")
    p.add_argument("--chrome-binary", help="Path to Chrome binary (optional)")
    p.add_argument("--profile-dir", help="Custom Chrome profile dir (optional)")
    # Timeout default increased to 600s
    p.add_argument("--timeout", type=int, default=1600, help="Overall run timeout (seconds)")
    p.add_argument("--page-delay", type=float, default=3.0,
                   help="Seconds to wait between pages after extraction (default: 3.0)")
    p.add_argument("-o","--out", default=str(OUT_JSON), help="Output JSON path")
    p.add_argument("--csv", help="Also write CSV to this path (e.g., reviews.csv)")
    return p.parse_args()

async def run_cli(args):
    global OUT_JSON
    OUT_JSON = Path(args.out)
    client = GlassdoorReviews(
        headless=args.headless,
        chrome_path=args.chrome_binary,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None
    )
    await client.scrape_reviews(
        args.url,
        pages=args.pages,
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
        log(f"[main][ERROR] Timed out after {a.timeout}s"); sys.exit(1)
    except Exception as e:
        import traceback
        log("[main][ERROR] Unhandled exception:")
        log("".join(traceback.format_exception(type(e), e, e.__traceback__))); sys.exit(1)
