from __future__ import annotations
import os, re, json, math, random, platform, ast
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features_exctract"
CONFIG_DIR   = ROOT / "config"
OUT_DIR      = ROOT / "out"

REVIEWS_PATH  = FEATURES_DIR / "combined_reviews.parquet"
GOALS_PATH    = CONFIG_DIR / "goal_dict.json"
CONFIG_JSON   = FEATURES_DIR / "config.json"

if not CONFIG_JSON.exists() and Path("/mnt/data/config.json").exists():
    CONFIG_JSON = Path("/mnt/data/config.json")

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("[auto] Using files:")
print("  reviews:", REVIEWS_PATH)
print("  goals:  ", GOALS_PATH)
print("  config: ", CONFIG_JSON)
print("  outdir: ", OUT_DIR, flush=True)


# ---------------- Goal configuration (5-goal version) ----------------
GOAL_NAMES = [
    "physiological",
    "self_protection",
    "affiliation",
    "status_esteem",
    "family_care",
]
GOAL_INDEX = {name: i for i, name in enumerate(GOAL_NAMES)}
N_GOALS = len(GOAL_NAMES)


# ---------------- Utils ----------------
def _need(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {label}: {p}")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def best_batch(default=64) -> int:
    if not torch.cuda.is_available():
        return default
    try:
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / (1024**3)
        if gb >= 20: return 128
        if gb >= 12: return 96
        if gb >= 8:  return 64
        return 48
    except Exception:
        return default


# --- Normalization & tokenization helpers (unified) ---
WS = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")

def norm_text_for_tokens(s: str) -> str:
    s = (s or "").replace("-", " ")
    return WS.sub(" ", s).strip().lower()

def tokenize(s: str) -> List[str]:
    s = norm_text_for_tokens(s)
    return [m.group(0).lower() for m in TOKEN_RE.finditer(s)]

_norm_space = re.compile(r"\s+")
def norm_term(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("-", " ")
    s = _norm_space.sub(" ", s)
    return s.lower().replace(" ", "_")

def ensure_variants(lst: List[str]) -> List[str]:
    return list({norm_term(t) for t in lst if str(t).strip()})


COUNTRY_CODES = {
    "us","uk","ca","au","in","de","fr","es","sg","ie","nl","se","ch","cn","jp","it","br","mx"
}

def _derive_company_id(source_file: str) -> str:
    """
    'reviews_accenture_us_clean.parquet' -> 'accenture'
    'reviews_boston_consulting_group_us_clean' -> 'boston_consulting_group'
    If no country present, just joins remaining parts.
    """
    stem = Path(str(source_file)).stem.lower()
    parts = [p for p in stem.split("_") if p not in {"reviews","review","clean",""}]
    if not parts:
        return "unknown"
    if parts[-1] in COUNTRY_CODES:
        parts.pop(-1)
    company = "_".join(parts) if parts else "unknown"
    return company


# ---------------- Loaders ----------------
def load_config(cpath: Path) -> dict:
    _need(cpath, "config.json")
    with open(cpath, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("negators", ["no","not","never","without","hardly","scarcely","lack","barely","rarely","seldom"])
    cfg.setdefault("offline_model", False)
    return cfg

def load_goals(gpath: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load the 5-goal dictionary from JSON.
    We keep short codes for compatibility:
      phys, selfprot, aff, stat, fam
    Guardrails are not used in this version.
    """
    _need(gpath, "goals dictionary json")
    raw = json.load(open(gpath, "r", encoding="utf-8"))
    mapping = {
        "physiological":   "phys",
        "self_protection": "selfprot",
        "affiliation":     "aff",
        "status_esteem":   "stat",
        "family_care":     "fam"
    }
    goals: Dict[str, Dict[str, List[str]]] = {}
    for k, v in raw.items():
        kk = mapping.get(k, k)
        goals[kk] = {
            "fulfillment": ensure_variants(v.get("fulfillment", [])),
            "hindrance":   ensure_variants(v.get("hindrance", [])),
        }
    return goals

def load_reviews(path: Path) -> pd.DataFrame:
    _need(path, "combined_reviews file")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    for col in ["review_id", "source_file"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing. Columns found: {list(df.columns)}")

    txt_col = "text_norm" if "text_norm" in df.columns else ("text_raw" if "text_raw" in df.columns else None)
    if txt_col is None:
        raise ValueError("Neither 'text_norm' nor 'text_raw' found in dataset.")
    df["text_clean"] = df[txt_col].fillna("").astype(str)

    def parse_token_cell(x):
        if isinstance(x, list):
            return tokenize(" ".join(map(str, x)))
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and "]" in s:
                try:
                    if "'" not in s and '"' not in s:
                        s = "[" + ",".join([f"'{t.strip()}'" for t in s.strip("[]").split(",")]) + "]"
                    vals = ast.literal_eval(s)
                    if isinstance(vals, list):
                        return tokenize(" ".join(map(str, vals)))
                except Exception:
                    pass
            return tokenize(s)
        return []

    if "tokens" in df.columns:
        df["tokens"] = df["tokens"].apply(parse_token_cell)
    else:
        df["tokens"] = df["text_clean"].map(tokenize)

    empty_mask = df["tokens"].apply(lambda z: not isinstance(z, list) or len(z) == 0)
    if empty_mask.any():
        df.loc[empty_mask, "tokens"] = df.loc[empty_mask, "text_clean"].map(tokenize)

    df["company_id"] = df["source_file"].map(_derive_company_id)

    if "date" not in df.columns:
        df["date"] = pd.NaT

    df["n_tokens"] = df["tokens"].apply(len)
    print(f"[load_reviews] rows={len(df)} companies≈{df['company_id'].nunique()} tokens_mean={df['n_tokens'].mean():.1f}")
    return df


# ---------------- Sentiment ----------------
class SentimentModel:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
                 offline=False, device=None, max_length=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=offline)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        self.model = self.model.to(self.device).eval()
        id2 = self.model.config.id2label
        lid = {v.lower(): k for k, v in id2.items()}
        self.i_neg = lid.get("negative", 0)
        self.i_neu = lid.get("neutral", 1)
        self.i_pos = lid.get("positive", 2)

    @torch.no_grad()
    def score(self, sentences: List[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(sentences), batch), desc="[sent] batches"):
            batch_s = sentences[i:i+batch]
            batch_s = [" ".join(s.split()[:60]) for s in batch_s]
            toks = self.tok(batch_s, padding=True, truncation=True,
                            max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**toks).logits, dim=-1).cpu().numpy()
            probs = probs[:, [self.i_neg, self.i_neu, self.i_pos]]
            out.append(probs)
        return np.vstack(out) if out else np.zeros((0,3))

    @staticmethod
    def to_scalar(p_neg: float, p_neu: float, p_pos: float) -> float:
        denom = max(1e-6, 1.0 - p_neu/2.0)
        return float(np.clip((p_pos - p_neg) / denom, -1.0, 1.0))


# ---------------- Sentence / negation helpers ----------------
_SENT_SPLIT = re.compile(r"[.!?]+")
def simple_sent_split(txt: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]

def find_phrase_positions(tokens: List[str], phrase_term: str) -> List[Tuple[int,int]]:
    """
    Exact consecutive-token match for underscore-joined phrases.
    """
    parts = phrase_term.split("_")
    n = len(parts); out = []
    if n == 1:
        for i, t in enumerate(tokens):
            if t == parts[0]:
                out.append((i, i+1))
    else:
        for i in range(len(tokens)-n+1):
            if tokens[i:i+n] == parts:
                out.append((i, i+n))
    return out

_SPACY_OK = False
try:
    import spacy
    _nlp = None
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

def detect_negated_spans(tokens: List[str], cfg_negators: List[str]) -> Set[int]:
    """
    Returns token indices considered negated.
    Uses spaCy if available; otherwise a simple negator-window fallback.
    """
    neg_idx: Set[int] = set()
    if _SPACY_OK:
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm", disable=["ner","textcat"])
            except Exception:
                _nlp = None
        if _nlp is not None:
            doc = _nlp(" ".join(tokens))
            for tok in doc:
                if tok.dep_.lower() == "neg":
                    head = tok.head
                    for w in head.subtree:
                        neg_idx.add(w.i)
            if neg_idx:
                return neg_idx

    neg_set = set(t.lower() for t in cfg_negators)
    for j, t in enumerate(tokens):
        if t in neg_set:
            for k in range(j+1, min(j+6, len(tokens))):
                neg_idx.add(k)
    return neg_idx


# ---------------- Option-C lexicon preparation ----------------
def build_lexicon(goals: Dict[str, Dict[str, List[str]]]) -> dict:
    """
    Builds efficient structures:
      - unigram sets (token)
      - bigram sets ((tok1,tok2))
      - long phrase lists (underscore terms, len>=3)
    For both fulfillment and hindrance.
    """
    lex = {}
    for g, d in goals.items():
        f_uni: Set[str] = set()
        f_bi: Set[Tuple[str,str]] = set()
        f_long: List[str] = []

        h_uni: Set[str] = set()
        h_bi: Set[Tuple[str,str]] = set()
        h_long: List[str] = []

        for term in d.get("fulfillment", []):
            parts = term.split("_")
            if len(parts) == 1:
                f_uni.add(parts[0])
            elif len(parts) == 2:
                f_bi.add((parts[0], parts[1]))
            else:
                f_long.append(term)

        for term in d.get("hindrance", []):
            parts = term.split("_")
            if len(parts) == 1:
                h_uni.add(parts[0])
            elif len(parts) == 2:
                h_bi.add((parts[0], parts[1]))
            else:
                h_long.append(term)

        lex[g] = {
            "F_uni": f_uni, "F_bi": f_bi, "F_long": f_long,
            "H_uni": h_uni, "H_bi": h_bi, "H_long": h_long,
        }
    return lex


def score_review_goals_option_c(
    tokens: List[str],
    neg_idx: Set[int],
    lex: dict,
    weights: dict
) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Compute raw evidence totals per goal:
      Fg[g] = fulfillment evidence
      Hg[g] = hindrance evidence

    - unigrams/bigrams: count occurrences via token scan
    - long phrases: count occurrences via exact consecutive-token spans
    - negation: if a match span touches neg_idx, flip polarity (F -> H or H -> F)
    """
    w_uni  = float(weights.get("w_uni", 1.0))
    w_bi   = float(weights.get("w_bi", 1.25))
    w_long = float(weights.get("w_long", 2.0))

    uni_counts = Counter(tokens)
    bi_counts = Counter(zip(tokens, tokens[1:])) if len(tokens) >= 2 else Counter()

    Fg: Dict[str, float] = {g: 0.0 for g in lex.keys()}
    Hg: Dict[str, float] = {g: 0.0 for g in lex.keys()}

    for g, gd in lex.items():
        # ----- unigrams -----
        for u in gd["F_uni"]:
            c = uni_counts.get(u, 0)
            if c <= 0:
                continue
            for i, tok in enumerate(tokens):
                if tok != u:
                    continue
                if i in neg_idx:
                    Hg[g] += w_uni
                else:
                    Fg[g] += w_uni

        for u in gd["H_uni"]:
            c = uni_counts.get(u, 0)
            if c <= 0:
                continue
            for i, tok in enumerate(tokens):
                if tok != u:
                    continue
                if i in neg_idx:
                    Fg[g] += w_uni
                else:
                    Hg[g] += w_uni

        # ----- bigrams -----
        for b in gd["F_bi"]:
            if bi_counts.get(b, 0) <= 0:
                continue
            a, b2 = b
            for i in range(len(tokens)-1):
                if tokens[i] == a and tokens[i+1] == b2:
                    if (i in neg_idx) or ((i+1) in neg_idx):
                        Hg[g] += w_bi
                    else:
                        Fg[g] += w_bi

        for b in gd["H_bi"]:
            if bi_counts.get(b, 0) <= 0:
                continue
            a, b2 = b
            for i in range(len(tokens)-1):
                if tokens[i] == a and tokens[i+1] == b2:
                    if (i in neg_idx) or ((i+1) in neg_idx):
                        Fg[g] += w_bi
                    else:
                        Hg[g] += w_bi

        # ----- long phrases (len>=3) -----
        for term in gd["F_long"]:
            spans = find_phrase_positions(tokens, term)
            for (L, R) in spans:
                if any(k in neg_idx for k in range(L, R)):
                    Hg[g] += w_long
                else:
                    Fg[g] += w_long

        for term in gd["H_long"]:
            spans = find_phrase_positions(tokens, term)
            for (L, R) in spans:
                if any(k in neg_idx for k in range(L, R)):
                    Fg[g] += w_long
                else:
                    Hg[g] += w_long

    return Fg, Hg


def balance_ratio(F: float, H: float, eps: float = 1e-6) -> float:
    """
    Recommended mixing normalization:
      R = (F - H) / (F + H + eps)   in [-1, 1]
    """
    return float((F - H) / (F + H + eps))


def sentiment_weight(Srev: float, sign: int) -> float:
    """
    Your existing sentiment coupling idea, preserved.
    Produces w in [lo, hi].
    """
    eps, tau, lo, hi = 0.08, 0.50, 0.3, 1.2
    w = 0.5 + 0.5 * sign * math.tanh(max(abs(Srev) - eps, 0.0) / tau)
    return float(np.clip(w, lo, hi))

def print_goal_coverage_summary(rev_out: pd.DataFrame):
    """
    Prints:
      - total reviews
      - how many hit at least one goal
      - how many hit zero goals
      - mean sentiment for each group
    """
    goal_cols = [c for c in rev_out.columns if c.startswith("G_final_")]

    total = len(rev_out)

    has_goal = (rev_out[goal_cols].abs().sum(axis=1) > 0)
    n_hit = int(has_goal.sum())
    n_zero = int((~has_goal).sum())

    mean_sent_all = float(rev_out["S_raw"].mean())
    mean_sent_hit = float(rev_out.loc[has_goal, "S_raw"].mean()) if n_hit > 0 else 0.0
    mean_sent_zero = float(rev_out.loc[~has_goal, "S_raw"].mean()) if n_zero > 0 else 0.0

    print("\n[goal coverage summary]")
    print(f"  total reviews           : {total}")
    print(f"  reviews with ≥1 goal hit: {n_hit} ({n_hit/total:.1%})")
    print(f"  reviews with 0 goals    : {n_zero} ({n_zero/total:.1%})")
    print(f"  mean sentiment (all)    : {mean_sent_all:+.3f}")
    print(f"  mean sentiment (≥1 goal): {mean_sent_hit:+.3f}")
    print(f"  mean sentiment (0 goals): {mean_sent_zero:+.3f}")


# ---------------- Main ----------------
def main():
    set_seed(42)

    cfg = load_config(CONFIG_JSON)
    reviews = load_reviews(REVIEWS_PATH)
    goals = load_goals(GOALS_PATH)

    lex = build_lexicon(goals)
    goal_keys = list(goals.keys())

    print(f"[load] reviews={len(reviews)} companies≈{reviews['company_id'].nunique()} goals={goal_keys}")

    sm = SentimentModel(
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        offline=bool(cfg.get("offline_model", False)),
        max_length=256
    )
    all_sents, ridx, slen = [], [], []
    for i, txt in enumerate(reviews["text_clean"].fillna("")):
        sents = simple_sent_split(txt) or [txt]
        for s in sents:
            all_sents.append(s); ridx.append(i); slen.append(max(1, len(s.split())))

    print(f"[sent] scoring {len(all_sents)} sentences…")
    probs = sm.score(all_sents, best_batch())
    s_scalar = [sm.to_scalar(p[0], p[1], p[2]) for p in probs]

    S = np.zeros(len(reviews), dtype=np.float32)
    bucket, weights = defaultdict(list), defaultdict(list)
    for v, i, L in zip(s_scalar, ridx, slen):
        bucket[i].append(v); weights[i].append(L)
    for i in range(len(reviews)):
        S[i] = float(np.average(bucket[i], weights=weights[i])) if bucket[i] else 0.0
    reviews["S_raw"] = S

    match_weights = {
        "w_uni": 1.0,
        "w_bi": 1.25,
        "w_long": 2.0,
    }

    outF = {g: [] for g in goal_keys}
    outH = {g: [] for g in goal_keys}
    outR = {g: [] for g in goal_keys}
    outW = {g: [] for g in goal_keys}
    outRw = {g: [] for g in goal_keys}

    negators_cfg = [t.lower() for t in cfg.get("negators", [])]

    for _, row in tqdm(list(reviews.iterrows()), desc="[goals] reviews"):
        toks: List[str] = row["tokens"]
        neg_idx = detect_negated_spans(toks, negators_cfg)

        Fg, Hg = score_review_goals_option_c(toks, neg_idx, lex, match_weights)

        Srev = float(row["S_raw"])

        for g in goal_keys:
            F = float(Fg.get(g, 0.0))
            H = float(Hg.get(g, 0.0))

            R = balance_ratio(F, H, eps=1e-6)
            sign = 1 if R > 0 else -1 if R < 0 else 0
            w = sentiment_weight(Srev, sign)

            Rw = float(np.clip(w * R, -1.0, 1.0))

            outF[g].append(F)
            outH[g].append(H)
            outR[g].append(R)
            outW[g].append(w)
            outRw[g].append(Rw)

    # ---------------- Outputs ----------------
    rev_out = pd.DataFrame({
        "review_id": reviews["review_id"],
        "company_id": reviews["company_id"],
        "date": reviews["date"],
        "n_tokens": reviews["n_tokens"],
        "S_raw": reviews["S_raw"]
    })

    for g in goal_keys:
        rev_out[f"F_raw_{g}"] = outF[g]
        rev_out[f"H_raw_{g}"] = outH[g]
        rev_out[f"G_ratio_{g}"] = outR[g]
        rev_out[f"w_sent_{g}"] = outW[g]
        rev_out[f"G_final_{g}"] = outRw[g]

    print("[aggregate] company metrics…")
    grp = rev_out.groupby("company_id", dropna=False)

    comp = grp["S_raw"].agg(["count","mean","std"]).reset_index().rename(
        columns={"count":"n_reviews","mean":"S_mean","std":"S_std"}
    )

    lamS = lamG = 80.0
    muS = float(comp["S_mean"].mean()) if len(comp) else 0.0
    eb = lambda m, n, lam, mu: (n/(n+lam))*m + (lam/(n+lam))*mu
    comp["S_smoothed"] = eb(comp["S_mean"], comp["n_reviews"].astype(float), lamS, muS)

    comp["SE"] = comp["S_std"] / np.sqrt(comp["n_reviews"].replace(0, np.nan))
    comp["S_CI_low"]  = comp["S_mean"] - 1.96 * comp["SE"]
    comp["S_CI_high"] = comp["S_mean"] + 1.96 * comp["SE"]

    pos_thr, neg_thr = 0.2, -0.2
    pos_share = grp.apply(lambda s: float(np.mean(s["S_raw"].to_numpy() > pos_thr))).rename("pos_share")
    neg_share = grp.apply(lambda s: float(np.mean(s["S_raw"].to_numpy() < neg_thr))).rename("neg_share")
    comp = comp.merge(pos_share, on="company_id").merge(neg_share, on="company_id")

    for g in goal_keys:
        gmeans = grp[[f"G_ratio_{g}", f"G_final_{g}"]].mean().reset_index()
        gmeans = gmeans.rename(columns={
            f"G_ratio_{g}": f"G_mean_ratio_{g}",
            f"G_final_{g}": f"G_mean_final_{g}"
        })
        comp = comp.merge(gmeans, on="company_id")

        muG = float(comp[f"G_mean_final_{g}"].mean()) if len(comp) else 0.0
        comp[f"G_smoothed_final_{g}"] = eb(
            comp[f"G_mean_final_{g}"],
            comp["n_reviews"].astype(float),
            lamG,
            muG
        )
    
    print_goal_coverage_summary(rev_out)


    # ----- Write outputs (CSV-only) -----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rev_path_csv  = OUT_DIR / "review_scores.csv"
    cmp_path_csv  = OUT_DIR / "company_scores.csv"
    per_dir       = OUT_DIR / "per_company"
    per_dir.mkdir(exist_ok=True)

    rev_out.to_csv(rev_path_csv, index=False)
    comp.to_csv(cmp_path_csv, index=False)

    import re as _re
    g_cols = [c for c in rev_out.columns if c.startswith("G_ratio_") or c.startswith("G_final_")]
    base_cols = ["review_id","company_id","S_raw"]
    keep_cols = base_cols + g_cols

    written_per_company = []
    for cid, sub in rev_out.groupby("company_id", dropna=False):
        safe = _re.sub(r"[^A-Za-z0-9_]+", "_", str(cid)).strip("_") or "unknown"
        (per_dir / f"{safe}.csv").write_text(sub[keep_cols].to_csv(index=False))
        written_per_company.append(str(per_dir / f"{safe}.csv"))

    run_meta = {
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "batch": best_batch(),
        "max_length": 256,
        "lexicon_match_weights": match_weights,
        "ratio_formula": "(F - H) / (F + H + eps)",
        "sentiment_weighting": {"eps": 0.08, "tau": 0.50, "lo": 0.3, "hi": 1.2},
        "offline_model": bool(cfg.get("offline_model", False)),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "output_files": {
            "review_csv": str(rev_path_csv),
            "company_csv": str(cmp_path_csv),
            "per_company_dir": str(per_dir),
            "per_company_files_count": len(written_per_company),
            "per_company_cols": keep_cols
        }
    }
    json.dump(
        {
            "N_reviews": int(len(reviews)),
            "companies": int(reviews["company_id"].nunique()),
            "goals": list(goal_keys),
            "columns_review": list(rev_out.columns),
            "columns_company": list(comp.columns),
            "run_meta": run_meta
        },
        open(OUT_DIR / "run_report.json", "w", encoding="utf-8"),
        indent=2
    )

    print(f"[done] CSVs:\n - {rev_path_csv}\n - {cmp_path_csv}\n - per-company (trimmed) in {per_dir}")


if __name__ == "__main__":
    main()
