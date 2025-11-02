import re, glob, html, os
import pandas as pd
from pathlib import Path


RAW_DIR = Path("C:/Users/csshl/Desktop/4994-Scrapper")
OUT_DIR = Path("C:/Users/csshl/Desktop/4994-Scrapper/cleaned_US")
OUT_DIR.mkdir(exist_ok=True)

MIN_LEN = 5              
KEEP_HYPHEN_APOS = True  


UA_PATHS = [
    Path(r"C:\Users\csshl\Desktop\4994-Scrapper\2025_Gaz_ua_national.txt"),
    Path("/mnt/data/2025_Gaz_ua_national.txt"),
]
SLDL_PATHS = [
    Path(r"C:\Users\csshl\Desktop\4994-Scrapper\2025_Gaz_sldl_national.txt"),
    Path("/mnt/data/2025_Gaz_sldl_national.txt"),
]


CANONICAL = {
    "review_id": ["review_id", "id", "reviewId", "reviewID"],
    "title": ["title", "summary", "headline"],
    "pros": ["pros", "pro", "positives"],
    "cons": ["cons", "con", "negatives"],
    "body": ["body", "review_text", "review", "details", "text"],
    "rating": ["rating", "overall_rating", "stars"],
    "date": ["date", "review_date", "timestamp", "created_at"],
    "job_title": ["job_title", "position", "role", "designation", "jobTitle", "author_job_title", "employee_title"],
    "company": ["company", "employer", "firm"], 
    "location": ["location", "city", "office_location", "site", "place"],
}

US_STATE_ABBR = {

    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

NON_US_KEYWORDS = {

    "india","canada","united kingdom","uk","england","scotland","wales","northern ireland",
    "australia","ireland","germany","france","spain","italy","netherlands","sweden","norway",
    "denmark","finland","switzerland","austria","belgium","poland","czech","japan","china",
    "singapore","hong kong","south korea","taiwan","thailand","philippines","malaysia",
    "mexico","brazil","chile","argentina","peru","colombia","south africa","nigeria",
    "uae","united arab emirates","saudi","qatar","kuwait","egypt","turkey","romania","portugal",
    "vietnam","indonesia","new zealand"
}

def find_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def robust_read_delim(path, sep="|"):

    return pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8", on_bad_lines="skip")

def load_gazetteers():
    ua_file = find_first_existing(UA_PATHS)
    sldl_file = find_first_existing(SLDL_PATHS)

    ua_df = pd.DataFrame()
    sldl_df = pd.DataFrame()
    if ua_file:
        ua_df = robust_read_delim(ua_file)
    if sldl_file:
        sldl_df = robust_read_delim(sldl_file)


    def norm(s):
        if not isinstance(s, str): return ""
        s = s.lower()
        s = re.sub(r"\burban area\b", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    us_name_set = set()

    if "NAME" in ua_df.columns:
        for name in ua_df["NAME"].dropna():
            base = norm(name)
            if base:
                us_name_set.add(base)
                for piece in re.split(r"\s*[,–—-]{1,2}\s*|\s*--\s*", name):
                    piece_n = norm(piece)
                    if piece_n:
                        us_name_set.add(piece_n)

    if "NAME" in sldl_df.columns:
        for name in sldl_df["NAME"].dropna():
            base = norm(name)
            if base:
                us_name_set.add(base)

    # Also add a few generic US markers
    us_name_set.update({
        "united states", "usa", "u s a", "u s", "us",
        "washington dc", "district of columbia"
    })

    return us_name_set

US_GAZ_NAMES = load_gazetteers()

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def standardize_columns(df):
    rename_map = {}
    for canon, aliases in CANONICAL.items():
        found = find_col(df, aliases)
        if found and found != canon:
            rename_map[found] = canon
    if rename_map:
        df = df.rename(columns=rename_map)

    for canon in CANONICAL.keys():
        if canon in ("company",):
            continue
        if canon not in df.columns:
            df[canon] = pd.NA
    return df

def file_slug_from_path(path):
    name = Path(path).stem
    m = re.search(r"reviews_([^\.]+)", name, flags=re.I)
    slug = m.group(1) if m else "unknown"
    return slug.lower().replace(" ", "_")

def normalize_date(series):
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    return parsed.dt.strftime("%Y-%m-%d")


TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile("["                             
                      u"\U0001F600-\U0001F64F"
                      u"\U0001F300-\U0001F5FF"
                      u"\U0001F680-\U0001F6FF"
                      u"\U0001F1E0-\U0001F1FF"
                      "]+", flags=re.UNICODE)

def clean_generic(s, keep_hyphen_apos=True):

    if not isinstance(s, str): return ""
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"http[s]?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = s.lower()
    if keep_hyphen_apos:
        s = re.sub(r"[^\w\s'-]", " ", s)
    else:
        s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def clean_title(s):

    if not isinstance(s, str): return ""
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def robust_read_csv(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            return pd.read_csv(path, dtype=str, encoding="latin-1", on_bad_lines="skip")

def norm_loc(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()

    s = re.sub(r"[^\w\s,]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

STATE_CODE_RE = re.compile(r",\s*([A-Z]{2})\b")

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

def infer_us_from_location(loc_raw: str):

    if not isinstance(loc_raw, str) or not loc_raw.strip():
        return False, "empty"
    s = loc_raw.strip()

    m = re.search(r",\s*([A-Za-z]{2})\b", s)
    if m and m.group(1).upper() in US_STATE_ABBR:
        return True, f"state_code:{m.group(1).upper()}"

    if ZIP_RE.search(s):
        return True, "zip"

    n = norm_loc(s)

    if n in US_GAZ_NAMES:
        return True, "gaz_full"

    pieces = {re.sub(r"\s{2,}", " ", p).strip() for p in re.split(r"[,\-–—/]", n)}
    pieces = {p for p in pieces if p}
    if any(p in US_GAZ_NAMES for p in pieces):
        return True, "gaz_piece"

    if any(k in n for k in NON_US_KEYWORDS):
        return False, "country_kw"

    return False, "fallback_nonus"

def process_file(path):
    df = robust_read_csv(path)
    original_rows = len(df)

    for col in ["source", "_source"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = standardize_columns(df)

    for col in ["source", "_source"]:
        if col in df.columns:
            df = df.drop(columns=[col])


    df["date"] = normalize_date(df["date"])

    before_dupes = len(df)
    if df["review_id"].notna().any():
        df = df.drop_duplicates(subset=["review_id"], keep="first")
    else:
        sig = (df["title"].fillna("") + "|" +
               df["pros"].fillna("") + "|" +
               df["cons"].fillna("") + "|" +
               df["body"].fillna(""))
        df = df.loc[~sig.duplicated(keep="first")]
    after_dupes = len(df)
    dropped_dupes = before_dupes - after_dupes

    mask_empty = (
        df["title"].fillna("").str.strip().eq("") &
        df["pros"].fillna("").str.strip().eq("") &
        df["cons"].fillna("").str.strip().eq("") &
        df["body"].fillna("").str.strip().eq("")
    )
    dropped_empty = int(mask_empty.sum())
    df = df.loc[~mask_empty].copy()

    df["title"] = df["title"].map(clean_title)
    for col in ["pros", "cons", "body"]:
        df[col] = df[col].map(lambda x: clean_generic(x, keep_hyphen_apos=KEEP_HYPHEN_APOS))

    df["text"] = (df["title"].fillna("") + " " +
                  df["pros"].fillna("") + " " +
                  df["cons"].fillna("") + " " +
                  df["body"].fillna("")).str.replace(r"\s{2,}", " ", regex=True).str.strip()

    dropped_short = 0
    if MIN_LEN is not None:
        short_mask = df["text"].str.len() < MIN_LEN
        dropped_short = int(short_mask.sum())
        df = df.loc[~short_mask].copy()

    if "location" not in df.columns:
        df["location"] = pd.NA

    us_flags = []
    reasons = []
    for v in df["location"].fillna(""):
        is_us, why = infer_us_from_location(v)
        us_flags.append("US" if is_us else "non-US")
        reasons.append(why)
    df["us_flag"] = us_flags           
    df["us_flag_reason"] = reasons     

    final_rows = len(df)

    drop_in_output = [c for c in ["body", "source", "_source"] if c in df.columns]
    if drop_in_output:
        df = df.drop(columns=drop_in_output)

    comp_slug = file_slug_from_path(path)
    out_path = OUT_DIR / f"reviews_{comp_slug}_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    return {
        "file": Path(path).name,
        "slug_from_file": comp_slug,
        "original_rows": original_rows,
        "dropped_dupes": dropped_dupes,
        "dropped_empty_text": dropped_empty,
        "dropped_too_short": dropped_short,
        "final_rows": final_rows,
        "output_file": str(out_path)
    }

files = sorted(glob.glob(str(RAW_DIR / "reviews_*.csv")))
print("Found:", len(files), "files")
summary = []

for fp in files:
    print("Processing:", Path(fp).name, "...")
    rec = process_file(fp)
    summary.append(rec)
    print("  -> saved:", rec["output_file"], "| kept:", rec["final_rows"])

summary_df = pd.DataFrame(summary).sort_values("slug_from_file").reset_index(drop=True)
summary_path = OUT_DIR / "phase1_2_cleaning_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8")

print("\nSummary saved to:", summary_path)
summary_df
