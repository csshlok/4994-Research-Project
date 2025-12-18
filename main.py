import sys
import time, random, json, urllib, logging, asyncio, re
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import datetime as dt
import selenium

from selenium import webdriver as wd
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException
from selenium.webdriver.chrome.service import Service

from urllib.parse import urlparse, parse_qs
from schema import SCHEMA

try:
    from pydoll_client import GlassdoorPydoll
    _pydoll_available = True
except Exception:
    _pydoll_available = False

start = time.time()

# ------------------ CLI Args ------------------
DEFAULT_URL = ('https://www.glassdoor.com/Overview/Working-at-'
               'Premise-Data-Corporation-EI_IE952471.11,35.htm')

parser = ArgumentParser()
parser.add_argument('-u', '--url', default=DEFAULT_URL,
                    help="Glassdoor landing page (Overview or Reviews)")
parser.add_argument('-f', '--file', default='glassdoor_ratings.csv',
                    help='Output CSV file')
parser.add_argument('--headless', action='store_true', help='Run Chrome headless')
parser.add_argument('--username', help='Glassdoor email')
parser.add_argument('-p', '--password', help='Glassdoor password')
parser.add_argument('-c', '--credentials', help='Path to JSON creds')
parser.add_argument('-l', '--limit', type=int, default=25,
                    help='Max reviews to scrape')
parser.add_argument('--start_from_url', action='store_true')
parser.add_argument('--max_date', type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
parser.add_argument('--min_date', type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
parser.add_argument('--manual_nav', action='store_true',
                    help='Interactive mode: YOU navigate pages; script only scrapes the current page.')

# -------- Engine selector + pydoll paging ----------
parser.add_argument('--engine', choices=['selenium', 'pydoll'], default='selenium',
                    help='Choose scraping engine. "pydoll" uses CDP + network capture.')
parser.add_argument('--pages', type=int, default=5,
                    help='(pydoll) Number of review pages to paginate through')

args = parser.parse_args()

if not args.start_from_url and (args.max_date or args.min_date):
    raise Exception('Invalid argument combination: No starting url passed, but max/min date specified.')
elif args.max_date and args.min_date:
    raise Exception('Invalid argument combination: Both min_date and max_date specified.')

if args.credentials:
    with open(args.credentials) as f:
        d = json.loads(f.read())
        args.username = d['username']
        args.password = d['password']
else:
    try:
        with open('secret.json') as f:
            d = json.loads(f.read())
            args.username = d['username']
            args.password = d['password']
    except FileNotFoundError:
        if args.engine == 'selenium':
            raise Exception("Please provide Glassdoor credentials via secret.json or CLI")

# ------------------ Logging ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d :%(filename)s(%(process)d) - %(message)s')
ch.setFormatter(formatter)

logging.getLogger('selenium').setLevel(logging.CRITICAL)

# ------------------ Human-like Helpers ------------------
def human_sleep(a=1.5, b=4.5):
    time.sleep(random.uniform(a, b))

def human_scroll_small():
    dy = random.randint(150, 400)
    browser.execute_script(f"window.scrollBy(0, {dy});")
    human_sleep(0.2, 0.7)

def human_hover(el):
    try:
        ActionChains(browser).move_to_element(el).pause(random.uniform(0.15, 0.35)).perform()
    except Exception:
        pass

def human_throttle_page_nav():
    human_sleep(6, 12)
    for _ in range(random.randint(1, 2)):
        human_scroll_small()

# ------------------ Cloudflare Handling ------------------
CF_VERIFY_TIMEOUT = 90
CF_POLL_INTERVAL  = 2
CF_MAX_PROMPTS    = 3
cf_prompt_count   = [0]

def detect_cloudflare(driver):
    """
    Returns True if current page looks like a Cloudflare/reCAPTCHA challenge.
    """
    try:
        src = driver.page_source.lower()
    except Exception:
        return False

    if ("verify you are human" in src
        or "help us protect glassdoor" in src
        or "please help us protect glassdoor" in src
        or "cf-103" in src):
        return True

    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for f in iframes:
            title = (f.get_attribute("title") or "").lower()
            s = (f.get_attribute("src") or "").lower()
            if "recaptcha" in title or "recaptcha" in s or "api2" in s:
                return True
    except Exception:
        pass

    try:
        if driver.find_elements(By.CSS_SELECTOR, "div.cf-browser-verification"):
            return True
    except Exception:
        pass

    return False

def wait_for_manual_cf_clear(driver,
                             prompt_msg="⚠️ Cloudflare challenge detected. Solve it in the browser, then press Enter to continue (Ctrl+C to abort).",
                             verify_timeout=CF_VERIFY_TIMEOUT,
                             poll_interval=CF_POLL_INTERVAL):
    """
    Pause automation so user can solve CF/recaptcha, then continue.
    """
    if getattr(args, "headless", False):
        raise RuntimeError("Cloudflare encountered in headless mode. Re-run without --headless to solve interactively.")
    if not sys.stdin or not sys.stdin.isatty():
        raise RuntimeError("No interactive terminal available to accept manual confirmation. Run from PowerShell/Terminal.")

    cf_prompt_count[0] += 1
    if cf_prompt_count[0] > CF_MAX_PROMPTS:
        raise RuntimeError(f"Cloudflare required manual solve {cf_prompt_count[0]} times. Aborting to avoid loops.")

    logger.warning(prompt_msg)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        raise RuntimeError("Manual solve aborted by user (EOF/KeyboardInterrupt).")

    start_t = time.time()
    while True:
        try:
            if not detect_cloudflare(driver):
                human_sleep(0.5, 1.0)
                logger.info("Cloudflare/reCAPTCHA no longer detected.")
                return
            try:
                if already_signed_in():
                    human_sleep(0.4, 1.0)
                    logger.info("Detected logged-in state after manual solve.")
                    return
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Error checking Cloudflare status: {e}")

        if time.time() - start_t > verify_timeout:
            try:
                cur_url = driver.current_url
            except Exception:
                cur_url = "<could not read url>"
            logger.error(f"Timeout waiting for Cloudflare to clear after manual solve. Current URL: {cur_url}")
            raise RuntimeError("Timed out waiting for Cloudflare to clear after manual solve.")

        time.sleep(poll_interval)

# ------------------ Browser (Selenium path) ------------------
PROFILE_DIR = r"C:/Users/csshl/Desktop/selenium/gd_profile"

def get_browser():
    logger.info('Configuring browser')
    chrome_options = wd.ChromeOptions()

    if not args.headless:
        chrome_options.add_argument(f"--user-data-dir={PROFILE_DIR}")
        chrome_options.add_argument("--profile-directory=Default")

    if args.headless:
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    chrome_options.page_load_strategy = "eager"

    ua = (
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(537,538)}.36 "
        f"(KHTML, like Gecko) Chrome/{random.randint(119,122)}.0.{random.randint(1000,5999)}.0 Safari/537.36"
    )
    chrome_options.add_argument(f"--user-agent={ua}")

    service = Service()
    br = wd.Chrome(service=service, options=chrome_options)
    br.set_page_load_timeout(45)
    br.set_script_timeout(45)

    br.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    })
    return br

def _switch_to_last_window(d):
    handles = d.window_handles
    if handles:
        d.switch_to.window(handles[-1])

def safe_get(d, url, expect=None, tries=3):
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            d.get(url)
            _switch_to_last_window(d)

            WebDriverWait(d, 30).until(
                lambda drv: drv.execute_script("return document.readyState") in ("interactive", "complete")
            )

            if detect_cloudflare(d):
                wait_for_manual_cf_clear(d)

            if expect:
                WebDriverWait(d, 30).until(EC.presence_of_element_located(expect))

            human_throttle_page_nav()
            return True

        except (TimeoutException, NoSuchWindowException) as e:
            last_exc = e
            logger.warning(f"safe_get attempt {attempt}/{tries} failed: {e}. Retrying...")
            try: _switch_to_last_window(d)
            except Exception: pass
            try: d.refresh()
            except Exception: pass
            human_sleep(1.5, 3.0)

    if last_exc:
        logger.exception("safe_get failed all attempts.")
        raise last_exc
    return False

# ------------------ Sorting / Dates ------------------
def verify_date_sorting():
    logger.info('Date limit specified, verifying date sorting')
    qs = parse_qs(urlparse(args.url).query)
    ascending = (qs.get('sort.ascending', ['false'])[0].lower() == 'true')

    if args.min_date and ascending:
        raise Exception('min_date requires DESCENDING sorting by date (sort.ascending=false).')
    if args.max_date and not ascending:
        raise Exception('max_date requires ASCENDING sorting by date (sort.ascending=true).')

# ------------------ Navigation ------------------
def on_reviews_page():
    url = browser.current_url.lower()
    if "/reviews" in url:
        return True
    try:
        tab = browser.find_element(By.CSS_SELECTOR, '#reviews[data-test="ei-nav-reviews-link"]')
        if tab.get_attribute("data-ui-selected") == "true":
            return True
    except Exception:
        pass
    return 'pageName":"ei-reviews' in browser.page_source

def navigate_to_reviews():
    logger.info("Navigating to company reviews")
    safe_get(browser, args.url, expect=(By.TAG_NAME, "body"))

    if detect_cloudflare(browser):
        wait_for_manual_cf_clear(browser)
    if on_reviews_page():
        logger.info("Already on a Reviews page")
        return True

    try:
        tab_link = WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#reviews[data-test="ei-nav-reviews-link"] a, a[data-test="ei-nav-reviews-link"]'))
        )
        browser.execute_script("arguments[0].scrollIntoView({block:'center'});", tab_link)
        try: tab_link.click()
        except Exception: browser.execute_script("arguments[0].click();", tab_link)
        human_throttle_page_nav()
        if detect_cloudflare(browser):
            wait_for_manual_cf_clear(browser)
        WebDriverWait(browser, 15).until(lambda d: on_reviews_page())
        return True
    except Exception:
        cur = browser.current_url
        if "/overview/" in cur.lower():
            reviews_url = cur.replace("/Overview/", "/Reviews/").replace("/overview/", "/reviews/").split("?")[0]
            logger.info(f"Jumping to {reviews_url}")
            safe_get(browser, reviews_url, expect=(By.TAG_NAME, "body"))
            return True
        return False

def get_current_page():
    logger.info('Getting current page number')
    candidates = [
        (By.CSS_SELECTOR, '[data-test="pagination-pages"] [aria-current="page"]'),
        (By.CSS_SELECTOR, 'nav[aria-label="Pagination"] [aria-current="page"]'),
        (By.CSS_SELECTOR, '.pageContainer .selected'),
    ]
    for by, sel in candidates:
        try:
            el = WebDriverWait(browser, 5).until(EC.presence_of_element_located((by, sel)))
            return int(el.text.strip())
        except Exception:
            pass
    raise TimeoutException("Could not determine current page")

def more_pages():
    next_candidates = [
        (By.CSS_SELECTOR, 'button[aria-label="Next"][aria-disabled="false"]'),
        (By.CSS_SELECTOR, 'a[aria-label="Next"]:not([aria-disabled="true"])'),
        (By.CSS_SELECTOR, '.nextButton:not(.disabled)'),
    ]
    for by, sel in next_candidates:
        try:
            el = browser.find_element(by, sel)
            return el.is_displayed()
        except Exception:
            continue
    return False

def go_to_next_page():
    logger.info(f'Going to page {page[0] + 1}')
    next_candidates = [
        (By.CSS_SELECTOR, 'button[aria-label="Next"][aria-disabled="false"]'),
        (By.CSS_SELECTOR, 'a[aria-label="Next"]:not([aria-disabled="true"])'),
        (By.CSS_SELECTOR, '.nextButton:not(.disabled)'),
    ]
    btn = None
    for by, sel in next_candidates:
        els = browser.find_elements(by, sel)
        if els:
            btn = els[0]; break
    if not btn:
        valid_page[0] = False
        return
    browser.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    human_hover(btn); human_sleep(0.2, 0.7)
    try:
        btn.click()
    except Exception:
        browser.execute_script("arguments[0].click();", btn)

    human_throttle_page_nav()
    if detect_cloudflare(browser):
        wait_for_manual_cf_clear(browser)
    page[0] += 1

# ------------------ Scraping (Selenium path) ------------------
def _first(locator_list):
    for by, sel in locator_list:
        els = browser.find_elements(by, sel)
        if els:
            return els[0]
    return None

# ---------- Your additional Selenium scrapers ----------
def scrape_overall_rating(_review):
    """
    Return overall rating as a STRING like "4.0".
    Falls back to "N/A" if not found.
    """
    locators = [
        (By.CSS_SELECTOR, '[data-test="rating-value"], [data-test="overallRating"]'),
        (By.XPATH, './/*[@aria-label and contains(@aria-label,"out of 5")]'),
        (By.XPATH, './/*[contains(@class,"rating") and (contains(text()," out of 5") or normalize-space(.)="5")]')
    ]

    def _num_from_text(s: str):
        if not s:
            return None
        m = re.search(r'([0-5](?:\.[0-9])?)', s)
        return m.group(1) if m else None

    for by, sel in locators:
        try:
            els = _review.find_elements(by, sel)
            for el in els:
                txt = None
                try:
                    txt = el.text
                except Exception:
                    txt = None
                if not txt:
                    for attr in ("aria-label", "title"):
                        try:
                            v = el.get_attribute(attr)
                            if v:
                                txt = v; break
                        except Exception:
                            pass
                val = _num_from_text((txt or "").strip())
                if val:
                    try:
                        return f"{float(val):.1f}"
                    except Exception:
                        return val
        except Exception:
            continue

    try:
        card_txt = _review.text
        val = _num_from_text(card_txt)
        if val:
            try:
                return f"{float(val):.1f}"
            except Exception:
                return val
    except Exception:
        pass

    return "N/A"

def _scrape_badge(review, label):
    """
    Detects badge status for 'Recommend', 'Business outlook', 'CEO approval'.
    Returns one of: "Yes", "No", "Neutral", "N/A"
    """
    try:
        el = review.find_element(
            By.XPATH,
            f'.//div[contains(@class,"rating-icon")][.//span[contains(text(),"{label}")]]'
        )
        classes = el.get_attribute("class") or ""

        if "positiveStyles" in classes:
            return "Yes"
        if "negativeStyles" in classes:
            return "No"
        if "neutralStyles" in classes:
            return "Neutral"

        try:
            svg = el.find_element(By.TAG_NAME, "svg").get_attribute("outerHTML").lower()
            if "path" in svg and "evenodd" in svg:
                return "No"
            if "<circle" in svg and "stroke" in svg:
                return "Neutral"
        except Exception:
            pass

        return "N/A"
    except Exception:
        return "N/A"

def scrape_recommends(review):
    return _scrape_badge(review, "Recommend")

def scrape_outlook(review):
    return _scrape_badge(review, "Business outlook")

def scrape_approve_ceo(review):
    return _scrape_badge(review, "CEO approval")

def _expand_show_more_in_text(_review):
    """
    If the review body is truncated, click 'Continue reading' / 'Show more'.
    Safe to call multiple times.
    """
    for by, sel in [
        (By.XPATH, './/button[contains(.,"Continue reading")]'),
        (By.XPATH, './/button[contains(.,"Show more")]'),
        (By.CSS_SELECTOR, 'button[data-test*="expand"], button[aria-expanded="false"]')
    ]:
        try:
            btns = _review.find_elements(by, sel)
            if btns:
                try:
                    btns[0].click()
                except Exception:
                    _review.parent.execute_script("arguments[0].click();", btns[0])
                time.sleep(0.1)
                break
        except Exception:
            pass

def scrape_pros(_review):
    """
    Returns the Pros text as a string, or 'N/A' if none found.
    """
    _expand_show_more_in_text(_review)
    locators = [
        (By.CSS_SELECTOR, '[data-test="review-text-PROS"]'),
        (By.XPATH, './/*[contains(@data-test,"PROS")]'),
        (By.XPATH, './/p[preceding-sibling::*[contains(normalize-space(.),"Pros")]][1]'),
    ]
    for by, sel in locators:
        try:
            els = _review.find_elements(by, sel)
            if els and (txt := els[0].text.strip()):
                return txt
        except Exception:
            continue
    return "N/A"

def scrape_cons(_review):
    """
    Returns the Cons text as a string, or 'N/A' if none found.
    """
    _expand_show_more_in_text(_review)
    locators = [
        (By.CSS_SELECTOR, '[data-test="review-text-CONS"]'),
        (By.XPATH, './/*[contains(@data-test,"CONS")]'),
        (By.XPATH, './/p[preceding-sibling::*[contains(normalize-space(.),"Cons")]][1]'),
    ]
    for by, sel in locators:
        try:
            els = _review.find_elements(by, sel)
            if els and (txt := els[0].text.strip()):
                return txt
        except Exception:
            continue
    return "N/A"

def scrape_advice(_review):
    """
    Returns the 'Advice to Management' text as a string, or 'N/A' if none found.
    """
    _expand_show_more_in_text(_review)
    locators = [
        (By.CSS_SELECTOR, '[data-test="review-text-ADVICE"]'),
        (By.XPATH, './/*[contains(@data-test,"ADVICE")]'),
        (By.XPATH, './/p[preceding-sibling::*[contains(normalize-space(.),"Advice")]][1]'),
        (By.XPATH, './/p[preceding-sibling::*[contains(normalize-space(.),"Advice to Management")]][1]'),
    ]
    for by, sel in locators:
        try:
            els = _review.find_elements(by, sel)
            if els and (txt := els[0].text.strip()):
                return txt
        except Exception:
            continue
    return "N/A"


# ================== SCRAPE DISPATCHER (Selenium DOM) ==================

# Small utilities
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else ""

def _digits(s: str) -> str | None:
    m = re.search(r"\d+", s or "")
    return m.group(0) if m else None

def _float_0_5(s: str) -> str | None:
    m = re.search(r"\b([0-5](?:\.[0-9])?)\b", s or "")
    return f"{float(m.group(1)):.1f}" if m else None

def _first_text(el, locators):
    for by, sel in locators:
        try:
            cand = el.find_elements(by, sel)
            if cand:
                txt = _norm(cand[0].text)
                if txt:
                    return txt
        except Exception:
            pass
    return ""

# ---- Field scrapers ----

def _scrape_date(review, author):
    locs = [
        (By.CSS_SELECTOR, '[data-test="review-date"]'),
        (By.CSS_SELECTOR, 'time[datetime]'),
        (By.XPATH, './/*[contains(@class,"date") and (contains(text(),",") or contains(text(),"20"))]'),
    ]
    txt = _first_text(review, locs)
    if not txt:
        txt = _norm(review.text)
    return txt or "N/A"

def _scrape_employee_title(review, author):
    locs = [
        (By.CSS_SELECTOR, '[data-test="reviewer-job-title"]'),
        (By.XPATH, './/*[contains(@class,"authorJobTitle") or contains(@class,"jobTitle")]'),
        (By.XPATH, './/span[contains(@class,"reviewer")]'),
    ]
    return _first_text(review, locs) or "N/A"

def _scrape_location(review, author):
    locs = [
        (By.CSS_SELECTOR, '[data-test="reviewer-location"]'),
        (By.XPATH, './/*[contains(@class,"location")]'),
        (By.XPATH, './/*[contains(text(),"Location")]/following::*[1]'),
    ]
    return _first_text(review, locs) or "N/A"

def _scrape_employee_status(review, author):
    locs = [
        (By.CSS_SELECTOR, '[data-test="reviewer-status"]'),
        (By.XPATH, './/*[contains(@class,"employmentStatus") or contains(text(),"Current Employee") or contains(text(),"Former Employee")]'),
    ]
    txt = _first_text(review, locs)
    if txt:
        m = re.search(r'(Current|Former)\s+Employee\b.*', txt, re.I)
        if m: return _norm(m.group(0))
        return txt
    return "N/A"

def _scrape_review_title(review, author):
    locs = [
        (By.CSS_SELECTOR, '[data-test="review-title"], h2[data-test], h2'),
        (By.XPATH, './/a[contains(@data-test,"review-title") or contains(@href,"/Reviews/")]|.//h2'),
    ]
    return _first_text(review, locs) or "N/A"

def _scrape_helpful(review, author):
    candidates = [
        (By.CSS_SELECTOR, '[data-test*="helpful"], button[aria-label*="Helpful"], [aria-label*="Helpful"]'),
        (By.XPATH, './/button[contains(.,"Helpful") or contains(@aria-label,"Helpful") or contains(@data-test,"helpful")]'),
    ]
    for by, sel in candidates:
        try:
            els = review.find_elements(by, sel)
            for el in els:
                text_pool = [el.text, el.get_attribute("aria-label") or "", el.get_attribute("title") or ""]
                for t in text_pool:
                    n = _digits(t)
                    if n is not None:
                        return int(n)
        except Exception:
            pass
    return 0

def _scrape_subrating(review, keywords: list[str], data_tests: list[str] = None):
    """
    Try to read a subrating like 'Work/Life Balance', 'Culture & Values', etc.
    Returns "x.y" string or "N/A".
    """
    data_tests = data_tests or []
    for key in data_tests:
        try:
            els = review.find_elements(By.CSS_SELECTOR, f'[data-test*="{key}"]')
            for el in els:
                txt = _norm(el.text) or _norm(el.get_attribute("aria-label") or "") or _norm(el.get_attribute("title") or "")
                val = _float_0_5(txt)
                if val: return val
        except Exception:
            pass

    for kw in keywords:
        try:
            els = review.find_elements(By.XPATH, f'.//*[contains(translate(.,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "{kw.lower()}")]')
            for lab in els:
                txt = _norm(lab.text) or _norm(lab.get_attribute("aria-label") or "")
                val = _float_0_5(txt)
                if val: return val
                siblings = lab.find_elements(By.XPATH, './/following::*[position()<=3]')
                for sib in siblings:
                    t2 = _norm(sib.text) or _norm(sib.get_attribute("aria-label") or "") or _norm(sib.get_attribute("title") or "")
                    val = _float_0_5(t2)
                    if val: return val
        except Exception:
            pass
    return "N/A"

def _scrape_rating_balance(review, author):
    return _scrape_subrating(
        review,
        keywords=["work/life balance", "work life balance", "work-life balance", "balance"],
        data_tests=["workLifeBalance", "rating-workLifeBalance"]
    )

def _scrape_rating_culture(review, author):
    return _scrape_subrating(
        review,
        keywords=["culture & values", "culture and values", "culture", "values"],
        data_tests=["cultureAndValues", "rating-cultureAndValues"]
    )

def _scrape_rating_career(review, author):
    return _scrape_subrating(
        review,
        keywords=["career opportunities", "career growth", "opportunities"],
        data_tests=["careerOpportunities", "rating-careerOpportunities"]
    )

def _scrape_rating_comp(review, author):
    return _scrape_subrating(
        review,
        keywords=["compensation & benefits", "compensation and benefits", "compensation", "benefits"],
        data_tests=["compensationAndBenefits", "rating-compensationAndBenefits"]
    )

def _scrape_rating_mgmt(review, author):
    return _scrape_subrating(
        review,
        keywords=["senior management", "management"],
        data_tests=["seniorManagement", "rating-seniorManagement"]
    )

# Dispatcher mapping to SCHEMA fields
_FIELD_FUNCS = {
    'date': _scrape_date,
    'employee_title': _scrape_employee_title,
    'location': _scrape_location,
    'employee_status': _scrape_employee_status,
    'review_title': _scrape_review_title,
    'helpful': _scrape_helpful,
    'pros': scrape_pros,
    'cons': scrape_cons,
    'advice_to_mgmt': scrape_advice,
    'rating_overall': lambda r,a: scrape_overall_rating(r),
    'rating_balance': _scrape_rating_balance,
    'rating_culture': _scrape_rating_culture,
    'rating_career': _scrape_rating_career,
    'rating_comp': _scrape_rating_comp,
    'rating_mgmt': _scrape_rating_mgmt,
    'recommends': scrape_recommends,
    'positive_outlook': scrape_outlook,
    'approves_of_CEO': scrape_approve_ceo,
}

def scrape(field: str, review, author=None):
    """
    Generic field dispatcher used by extract_review().
    review: Selenium WebElement representing the review card/article.
    author: optional WebElement for the reviewer metadata area.
    """
    fn = _FIELD_FUNCS.get(field)
    if not fn:
        return np.nan
    try:
        return fn(review, author)
    except Exception:
        return np.nan
# ================== END SCRAPE DISPATCHER ==================

# ---------- end extra scrapers ----------

def extract_review(review):
    try:
        author = review.find_element(By.CSS_SELECTOR, '[data-test="reviewer"]')
    except Exception:
        author = None
    row = {}
    for field in SCHEMA:
        try:
            row[field] = scrape(field, review, author)
        except Exception:
            row[field] = np.nan

    try:
        if args.max_date or args.min_date:
            rv_dt = row.get('review_date')
            if isinstance(rv_dt, str):
                for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
                    try:
                        rv_dt = dt.datetime.strptime(rv_dt, fmt)
                        break
                    except Exception:
                        pass
            if isinstance(rv_dt, dt.datetime):
                if args.min_date and rv_dt < args.min_date:
                    date_limit_reached[0] = True
    except Exception:
        pass

    return row

def extract_from_page():
    logger.info(f'Extracting reviews from page {page[0]}')
    res = pd.DataFrame([], columns=SCHEMA)

    for _ in range(random.randint(2, 4)):
        human_scroll_small()
    if detect_cloudflare(browser):
        wait_for_manual_cf_clear(browser)

    selectors = [
        (By.CSS_SELECTOR, '[data-test="review"], [data-test="EmpReview"], article[data-test*="review"]'),
        (By.CLASS_NAME, 'empReview')
    ]
    reviews = []
    for by, sel in selectors:
        reviews = browser.find_elements(by, sel)
        if reviews:
            break

    logger.info(f"Found {len(reviews)} reviews on page {page[0]}")

    if not reviews:
        browser.refresh(); human_throttle_page_nav()
        if detect_cloudflare(browser):
            wait_for_manual_cf_clear(browser)
        for by, sel in selectors:
            reviews = browser.find_elements(by, sel)
            if reviews:
                break
        if not reviews:
            valid_page[0] = False
            return res

    for review in reviews:
        try:
            browser.execute_script("arguments[0].scrollIntoView({block:'center'});", review)
        except Exception:
            pass
        human_hover(review); human_sleep(0.3, 1.1)
        data = extract_review(review)
        res.loc[idx[0]] = data
        idx[0] += 1
    return res

# ------------------ Auth (Selenium path) ------------------
def sign_in():
    logger.info(f'Signing in to {args.username}')
    login_url = 'https://www.glassdoor.com/member/profile/login'
    safe_get(browser, login_url)

    wait = WebDriverWait(browser, 40)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "form")))

    email = wait.until(lambda d: _first([
        (By.CSS_SELECTOR, 'input[data-test="emailInput-input"]'),
        (By.ID, 'inlineUserEmail'),
        (By.NAME, 'username'),
        (By.CSS_SELECTOR, 'input[type="email"]'),
    ]))
    email.clear(); email.send_keys(args.username)

    cont = wait.until(lambda d: _first([
        (By.XPATH, '//button[contains(.,"Continue with email")]'),
        (By.CSS_SELECTOR, 'button[data-test="emailButton"]'),
        (By.XPATH, '//button[@type="submit"]'),
    ]))
    cont.click()

    _switch_to_last_window(browser)

    pwd = wait.until(lambda d: _first([
        (By.CSS_SELECTOR, 'input[data-test="passwordInput-input"]'),
        (By.ID, 'inlineUserPassword'),
        (By.NAME, 'password'),
        (By.CSS_SELECTOR, 'input[type="password"]'),
    ]))
    pwd.clear(); pwd.send_keys(args.password)

    signin = wait.until(lambda d: _first([
        (By.XPATH, '//button[contains(.,"Sign in") or contains(.,"Sign In")]'),
        (By.XPATH, '//button[@type="submit"]'),
    ]))
    signin.click()

    safe_get(browser, args.url, expect=(By.TAG_NAME, "body"))

def already_signed_in():
    src = browser.page_source.lower()
    return ("global-nav" in src) and ("sign in" not in src)

# ------------------ Manual navigation loop (Selenium path) ------------------
def manual_navigation_loop(res):
    """
    You control navigation (solve CF, open reviews, click next).
    Script only scrapes current page after you press Enter.
    """
    logger.info("Manual navigation mode: open the desired Reviews page in the browser window, then press Enter here.")
    if getattr(args, "headless", False):
        raise RuntimeError("--manual_nav requires a visible (non-headless) browser window.")

    while True:
        input("When the Reviews page is fully loaded in the browser, press Enter to scrape this page (Ctrl+C to finish)... ")

        if detect_cloudflare(browser):
            print("\nCloudflare page still detected. Solve it in the browser, then press Enter again.\n")
            continue

        try:
            page_df = extract_from_page()
            remaining = args.limit - len(res)
            if remaining <= 0:
                break
            if len(page_df) > remaining:
                page_df = page_df.iloc[:remaining]
            res = pd.concat([res, page_df], ignore_index=True)
            logger.info(f"Collected {len(res)}/{args.limit} reviews so far.")
        except Exception:
            logger.exception("Failed to extract from the current page. Fix in the browser and press Enter to retry.")
            continue

        if len(res) >= args.limit:
            break

        print("\nNow YOU click the site’s Next (or change filters/date) in the browser.")
        print("After the page loads, press Enter here to scrape the next page. (Ctrl+C to finish early.)")

    return res

# ------------------ Pydoll integration (engine + CSV export) ------------------
def run_with_pydoll():
    if not _pydoll_available:
        raise RuntimeError('pydoll engine selected but "pydoll_client.py" or pydoll is not available. '
                           'Install pydoll and ensure pydoll_client.py is present.')

    logger.info(f'[pydoll] Starting scrape from {args.url} for ~{args.pages} pages')
    client = GlassdoorPydoll(headless=args.headless)
    reviews = asyncio.run(client.scrape_reviews(args.url, pages=args.pages))

    if not reviews:
        df = pd.DataFrame([])
    else:
        df = pd.json_normalize(reviews)

    if args.limit and len(df) > args.limit:
        df = df.iloc[:args.limit]

    if len(df) > 0:
        common = [c for c in SCHEMA if c in df.columns]
        others = [c for c in df.columns if c not in common]
        ordered_cols = common + others if common else df.columns.tolist()
        df = df[ordered_cols]

    df.to_csv(args.file, index=False, encoding='utf-8')
    logger.info(f'[pydoll] Wrote {len(df)} rows to {args.file}')
    return

# ------------------ Scrape Dispatcher ------------------
browser = None
page = [1]; idx = [0]; date_limit_reached = [False]; valid_page = [True]

def main():
    if args.engine == 'pydoll':
        return run_with_pydoll()

    logger.info(f"Scraping up to {args.limit} reviews (engine=selenium)")
    global browser
    browser = get_browser()
    res = pd.DataFrame([], columns=SCHEMA)

    if args.manual_nav and args.headless:
        raise RuntimeError("--manual_nav cannot be used with --headless.")

    if not already_signed_in():
        try:
            sign_in()
        except RuntimeError as e:
            if "Cloudflare" in str(e):
                wait_for_manual_cf_clear(browser)
            else:
                raise
        if detect_cloudflare(browser):
            wait_for_manual_cf_clear(browser)

    if args.manual_nav:
        res = manual_navigation_loop(res)
    else:
        if not args.start_from_url:
            reviews_exist = navigate_to_reviews()
            if not reviews_exist:
                logger.error("Could not reach a Reviews page.")
                return
        elif args.max_date or args.min_date:
            verify_date_sorting()
            safe_get(browser, args.url, expect=(By.TAG_NAME, "body"))
            page[0] = get_current_page()
            logger.info(f'Starting from page {page[0]:,}.')
            time.sleep(1)
        else:
            safe_get(browser, args.url, expect=(By.TAG_NAME, "body"))
            page[0] = get_current_page()
            logger.info(f'Starting from page {page[0]:,}.')
            time.sleep(1)

        reviews_df = extract_from_page()
        res = pd.concat([res, reviews_df], ignore_index=True)

        while more_pages() and len(res) < args.limit and not date_limit_reached[0] and valid_page[0]:
            go_to_next_page()
            try:
                reviews_df = extract_from_page()
                res = pd.concat([res, reviews_df], ignore_index=True)
            except Exception:
                logger.exception("Error extracting from page; stopping pagination.")
                break

    logger.info(f'Writing {len(res)} reviews to file {args.file}')
    res.to_csv(args.file, index=False, encoding='utf-8')

    end = time.time()
    logger.info(f'Finished in {end - start} seconds')

if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            if browser:
                browser.quit()
        except Exception:
            pass
