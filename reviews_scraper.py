# reviews_scraper.py
import asyncio, json, re, inspect, sys, random, argparse, time
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
        sig = ["pros","cons","headline","reviewbody","overallrating","rating","jobtitle","createddatetime"]
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
            "title": r.get("headline") or r.get("title"),
            "body": r.get("reviewBody") or r.get("body") or r.get("text"),
            "pros": r.get("pros") or r.get("reviewPros"),
            "cons": r.get("cons") or r.get("reviewCons"),
            "rating": r.get("overallRating") or r.get("ratingOverall") or r.get("rating"),
            "date": r.get("reviewDate") or r.get("createdDateTime") or r.get("time"),
            "role": r.get("jobTitle") or r.get("authorJobTitle"),
            "location": r.get("location") or r.get("reviewerLocation"),
            "employmentStatus": r.get("employmentStatus"),
            "_source": "apollo",
        }
        k = x.get("review_id") or (x.get("title"), x.get("date"))
        if k and k not in seen: seen.add(k); out.append(x)
    return out

def _extract_next_data_reviews(html: str) -> List[Dict[str, Any]]:
    if not html: return []
    m = re.search(r'__NEXT_DATA__"\s*type="application/json">\s*({.+?})\s*</script>', html, flags=re.S|re.I)
    if not m: return []
    try:
        data = json.loads(m.group(1))
    except Exception as e:
        log(f"[next] JSON decode failed: {e}")
        return []
    cand: List[Dict[str, Any]] = []
    def walk(o):
        if isinstance(o, dict):
            keys = {k.lower() for k in o.keys()}
            if "pros" in keys or "cons" in keys: cand.append(o)
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
        k = x.get("review_id") or (x.get("title"), x.get("date"))
        if k and k not in seen: seen.add(k); out.append(x)
    return out

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
        # Non-blocking informational check
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
        """
        Unwrap driver-returned JS objects into a plain URL string.
        Handles nested {"result": {...}}, {"value": "..."} shells and extracts URLs from reprs.
        """
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

        # If not on Reviews tab, synthesize Reviews URL
        try:
            is_reviews = await self._eval_js(tab, "(() => location.pathname.toLowerCase().includes('/reviews'))();")
        except Exception:
            is_reviews = False
        if not is_reviews:
            import re as _re
            base = _re.sub(r'(?:_P\d+)?\.htm.*$', '', url).rstrip('/')
            await tab.go_to(base + "/Reviews.htm")

        # Ensure English param
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
    def _canonize_reviews_base(self, url: str) -> (str, dict):
        """
        Returns (base_without_suffix_htm, query_params_dict)
        such that page 1 is base + ".htm", page n>=2 is base + f"_P{n}.htm"
        """
        url = self._unwrap_url_obj(url)
        base = re.sub(r'(_P\d+)?\.htm.*$', '', url)  # strip existing _P# and .htm and anything after
        # preserve/normalize query params
        qs_match = re.search(r'\?(.+)$', url)
        qd = dict(parse_qsl(qs_match.group(1), keep_blank_values=True)) if qs_match else {}
        # force English
        qd['filter.iso3Language'] = 'eng'
        return base, qd

    def _page_url(self, base: str, qd: dict, page_idx: int) -> str:
        """Builds: base + .htm (p1) or _P{n}.htm (p>=2) + ?query"""
        path = f"{base}.htm" if page_idx == 1 else f"{base}_P{page_idx}.htm"
        qs = urlencode(qd, doseq=True)
        return f"{path}?{qs}" if qs else path

    # ---------------- main (JSON-only) ----------------
    async def scrape_reviews(self, company_url: str, pages: int = 3) -> List[Dict[str, Any]]:
        log("[run] Launching Chrome…")
        async with Chrome(options=self.opts) as browser:
            tab = await browser.start(); log("[run] Tab started")

            await self._goto_reviews_tab(tab, company_url)

            # Optional: request reverse-date sort if not present (harmless if ignored)
            try:
                changed = await self._eval_js(tab, """
                  (() => {
                    const u = new URL(location.href);
                    if (!u.searchParams.get('sort.sortType')) {
                      u.searchParams.set('sort.sortType','RD');
                      u.searchParams.set('sort.ascending','false');
                      location.replace(u.toString());
                      return true;
                    }
                    return false;
                  })();
                """)
                if changed: await self._sleep_human(0.7, 1.1)
            except Exception:
                pass

            cur_url = await self._current_href(tab)
            base, qd = self._canonize_reviews_base(cur_url)
            base = self._unwrap_url_obj(base)
            log(f"[nav] Base resolved: {base}.htm  | pages={pages}")

            all_rows: List[Dict[str, Any]] = []
            prev_ids: set = set()

            for page in range(1, max(1, pages)+1):
                # Navigate deterministically
                target = self._page_url(base, qd, page)
                target = self._unwrap_url_obj(target)
                log(f"[page] {page}/{pages} → {target}")
                await tab.go_to(target)
                await self._sleep_human(0.4, 0.8)
                await self._note_challenge(tab)
                landed = await self._current_href(tab)
                log(f"[nav] Landed: {landed}")

                # Light poke to encourage hydration (non-essential)
                try: await self._eval_js(tab, "window.scrollBy(0, 400);")
                except Exception: pass
                await self._sleep_human(0.2, 0.5)

                html = await self._get_page_source(tab)
                ids_now = await self._extract_ids_from_html(html)
                if page > 1 and ids_now and ids_now == prev_ids:
                    log("[warn] JSON IDs unchanged; page may not have advanced (rate-limit/CF). Continuing.")
                else:
                    prev_ids = ids_now

                rows = _extract_apollo_state_reviews(html) or _extract_next_data_reviews(html)
                log(f"[json] Extracted {len(rows)} objects on page {page}")
                all_rows.extend(rows)

            # de-dupe & normalize (keep __typename entries intact)
            dedup, seen = [], set()
            for r in all_rows:
                rating = r.get("rating")
                if isinstance(rating, str):
                    m = re.search(r'\d+(?:\.\d+)?', rating)
                    r["rating"] = float(m.group(0)) if m else None
                key = r.get("review_id") or (r.get("title"), r.get("date"), r.get("role"))
                if key not in seen:
                    seen.add(key); dedup.append(r)

            OUT_JSON.write_text(json.dumps(dedup, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"[run] Saved {len(dedup)} reviews → {OUT_JSON.resolve()}")
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
    p.add_argument("--timeout", type=int, default=240, help="Overall run timeout (seconds)")
    p.add_argument("-o","--out", default=str(OUT_JSON), help="Output JSON path")
    return p.parse_args()

async def run_cli(args):
    global OUT_JSON
    OUT_JSON = Path(args.out)
    client = GlassdoorReviews(
        headless=args.headless,
        chrome_path=args.chrome_binary,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None
    )
    await client.scrape_reviews(args.url, pages=args.pages)

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
