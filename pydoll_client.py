# pydoll_client.py
import asyncio, json, re, inspect, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions

PROFILE_DIR = Path("chrome-profile")
OUT_JSON    = Path("glassdoor_reviews.json")
DEBUG_HTML  = Path("glassdoor_debug.html")


def log(msg: str):
    print(msg, flush=True)


# --------------------------
# Apollo parser
# --------------------------
def _extract_apollo_state_reviews(html: str) -> List[Dict[str, Any]]:
    if not html:
        return []

    m = re.search(r'apolloState"\s*:\s*({.+?})\s*}\s*;', html, flags=re.S)
    if not m:
        return []

    try:
        apollo = json.loads(m.group(1))
    except Exception as e:
        log(f"[apolloState] JSON decode failed: {e}")
        return []

    reviews_raw: List[Dict[str, Any]] = []

    def _looks_like_review(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        keys = {k.lower() for k in d.keys()}
        signals = ["pros", "cons", "headline", "reviewbody", "overallrating", "rating"]
        return sum(1 for s in signals if any(s in k for k in keys)) >= 2

    def walk(o):
        if isinstance(o, dict):
            if _looks_like_review(o):
                reviews_raw.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(apollo)

    if reviews_raw:
        log(f"[debug] Collected {len(reviews_raw)} raw review-like objects")
        log(f"[debug] Sample keys: {list(reviews_raw[0].keys())[:12]}")

    norm: List[Dict[str, Any]] = []
    for r in reviews_raw:
        norm.append({
            "review_id": r.get("reviewId") or r.get("id"),
            "title": r.get("headline") or r.get("title"),
            "body": r.get("reviewBody") or r.get("body") or r.get("text"),
            "pros": r.get("pros") or r.get("reviewPros"),
            "cons": r.get("cons") or r.get("reviewCons"),
            "rating": r.get("overallRating") or r.get("ratingOverall") or r.get("rating"),
            "date": r.get("reviewDate") or r.get("createdDateTime") or r.get("time"),
            "role": r.get("jobTitle") or r.get("authorJobTitle"),
            "location": r.get("location") or r.get("reviewerLocation"),
        })

    # Deduplicate
    seen, out = set(), []
    for x in norm:
        k = x.get("review_id") or (x.get("title"), x.get("date"))
        if k in seen:
            continue
        seen.add(k)
        out.append(x)

    return out


class GlassdoorPydoll:
    def __init__(self, headless: bool = False, chrome_path: Optional[str] = None):
        log("[init] Building Chromium options")
        opts = ChromiumOptions()
        # keep only what we must add manually to avoid collisions with Pydoll defaults
        opts.add_argument(f"--user-data-dir={PROFILE_DIR.resolve()}")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        if chrome_path:
            try:
                opts.set_binary(chrome_path)
                log(f"[init] Using chrome binary: {chrome_path}")
            except Exception as e:
                log(f"[init] set_binary failed: {e}")
        if headless:
            opts.add_argument("--headless")
        self.opts = opts

    async def _eval_js(self, tab, script: str):
        return await tab.execute_script(script)

    async def _get_page_source(self, tab) -> str:
        attr = getattr(tab, "page_source", None)
        if attr is not None:
            if isinstance(attr, str):
                return attr
            if inspect.iscoroutine(attr):
                return await attr
            if callable(attr):
                res = attr()
                if inspect.iscoroutine(res):
                    return await res
                if isinstance(res, str):
                    return res

        get_ps = getattr(tab, "get_page_source", None)
        if callable(get_ps):
            res = get_ps()
            if inspect.iscoroutine(res):
                return await res
            if isinstance(res, str):
                return res

        for name in ("html", "content", "outer_html"):
            v = getattr(tab, name, None)
            if isinstance(v, str):
                return v
            if inspect.iscoroutine(v):
                return await v
            if callable(v):
                res = v()
                if inspect.iscoroutine(res):
                    return await res
                if isinstance(res, str):
                    return res

        try:
            return await self._eval_js(tab, "document.documentElement.outerHTML")
        except Exception:
            return ""

    async def _scrape_reviews_state(self, tab) -> List[Dict[str, Any]]:
        html = await self._get_page_source(tab)
        if not html:
            log("[apolloState] Empty page source")
            return []
        reviews = _extract_apollo_state_reviews(html)
        if reviews:
            log(f"[apolloState] Found {len(reviews)} normalized reviews")
        else:
            DEBUG_HTML.write_text(html or "", encoding="utf-8")
            log(f"[debug] 0 reviews parsed — wrote {DEBUG_HTML.resolve()}")
        return reviews

    async def _scrape_reviews_dom(self, tab) -> List[Dict[str, Any]]:
        js = r"""
        (() => {
          const cards = document.querySelectorAll('div[data-test="review-details-container"]');
          const out = [];
          cards.forEach(c => {
            const getText = (sel) => {
              const el = c.querySelector(sel);
              return el ? el.innerText.trim() : "";
            };
            const tagNodes = c.querySelectorAll('div[data-test="review-avatar-tag"] div.text-with-icon_LabelContainer__s0l4C');
            const tags = Array.from(tagNodes).map(el => el.innerText.trim());

            out.push({
              rating:   getText('span[data-test="review-rating-label"]'),
              date:     getText('span[class*="reviewDate"]') || getText('time'),
              title:    getText('h3[data-test="review-details-title"]') || getText('h3'),
              role:     getText('span[data-test="review-avatar-label"]'),
              status:   tags[0] || "",
              location: tags[1] || "",
              pros:     getText('span[data-test="review-text-PROS"]'),
              cons:     getText('span[data-test="review-text-CONS"]')
            });
          });
          return out;
        })();
        """
        out = await self._eval_js(tab, js)
        if out:
            log(f"[dom] Found {len(out)} reviews")
        return out if isinstance(out, list) else []

    async def scrape_reviews(self, company_url: str) -> List[Dict[str, Any]]:
        log(f"[run] Launching Chrome… URL: {company_url}")
        async with Chrome(options=self.opts) as browser:
            tab = await browser.start()
            log("[run] Tab started")
            await tab.go_to(company_url)
            log("[run] Navigation complete")

            # Apollo first
            reviews = await self._scrape_reviews_state(tab)

            # DOM fallback
            if not reviews:
                log("[run] Falling back to DOM scraping")
                reviews = await self._scrape_reviews_dom(tab)

            OUT_JSON.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"[run] Saved {len(reviews)} reviews → {OUT_JSON.resolve()}")
            return reviews


# Demo
async def _demo():
    url = "https://www.glassdoor.com/Reviews/Tata-Consultancy-Services-Reviews-E13461.htm?filter.iso3Language=eng"
    client = GlassdoorPydoll(headless=False)
    await client.scrape_reviews(url)


if __name__ == "__main__":
    log("[main] Starting pydoll_client.py")
    try:
        asyncio.run(asyncio.wait_for(_demo(), timeout=180))
        log("[main] Done")
    except asyncio.TimeoutError:
        log("[main][ERROR] Timed out after 180s")
        sys.exit(1)
    except Exception as e:
        import traceback
        log("[main][ERROR] Unhandled exception:")
        log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(1)
