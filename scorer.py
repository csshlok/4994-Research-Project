from __future__ import annotations
import os, re, json, math, random, platform, ast
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy import sparse

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features_exctract"
CONFIG_DIR   = ROOT / "config"
OUT_DIR      = ROOT / "out"

REVIEWS_PATH  = FEATURES_DIR / "combined_reviews.parquet"
VOCAB_PATH    = FEATURES_DIR / "tfidf_vocab.json"
TFIDF_NPZ     = FEATURES_DIR / "tfidf_reviews.npz"
GOALS_PATH    = CONFIG_DIR / "goal_dict.json"
CONFIG_JSON   = FEATURES_DIR / "config.json"

# allow fallbacks
if not CONFIG_JSON.exists() and Path("/mnt/data/config.json").exists():
    CONFIG_JSON = Path("/mnt/data/config.json")
if not VOCAB_PATH.exists() and Path("/mnt/data/tfidf_vocab.json").exists():
    VOCAB_PATH = Path("/mnt/data/tfidf_vocab.json")
if not TFIDF_NPZ.exists() and Path("/mnt/data/tfidf_reviews.npz").exists():
    TFIDF_NPZ = Path("/mnt/data/tfidf_reviews.npz")

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("[auto] Using files:")
print("  reviews:", REVIEWS_PATH)
print("  vocab:  ", VOCAB_PATH)
print("  tfidf:  ", TFIDF_NPZ)
print("  goals:  ", GOALS_PATH)
print("  config: ", CONFIG_JSON)
print("  outdir: ", OUT_DIR, flush=True)

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
TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")  # word or word'word

def norm_text_for_tokens(s: str) -> str:
    # Hyphens → spaces so "work-life" => "work life"
    s = (s or "").replace("-", " ")
    return WS.sub(" ", s).strip().lower()

def tokenize(s: str) -> List[str]:
    s = norm_text_for_tokens(s)
    return [m.group(0).lower() for m in TOKEN_RE.finditer(s)]

_norm_space = re.compile(r"\s+")
def norm_term(s: str) -> str:
    # Normalize goal dictionary entries to underscore-joined phrases
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
        parts.pop(-1)  # drop trailing country code
    company = "_".join(parts) if parts else "unknown"
    return company

# ---------------- Loaders ----------------
def load_config(cpath: Path) -> dict:
    _need(cpath, "config.json")
    with open(cpath, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("negators", ["no","not","never","without","hardly","scarcely","lack","barely","rarely","seldom"])
    cfg.setdefault("ngram_range", [1,2])
    cfg.setdefault("stop_extra", [])
    cfg.setdefault("offline_model", False)
    return cfg

def load_goals(gpath: Path) -> Dict[str, Dict[str, List[str]]]:
    _need(gpath, "goals dictionary json")
    raw = json.load(open(gpath, "r", encoding="utf-8"))
    mapping = {
        "physiological":"phys","self_protection":"selfprot","affiliation":"aff",
        "status_esteem":"stat","mate_acquisition":"m_acq","mate_retention":"m_ret","family_care":"fam"
    }
    goals = {}
    for k, v in raw.items():
        kk = mapping.get(k, k)
        goals[kk] = {
            "fulfillment": ensure_variants(v.get("fulfillment", [])),
            "hindrance":   ensure_variants(v.get("hindrance", [])),
            "guardrails":  ensure_variants(v.get("guardrails", [])),
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
        # Rehydrate stringified lists and re-tokenize uniformly
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

    # Fallback rebuild for empty
    empty_mask = df["tokens"].apply(lambda z: not isinstance(z, list) or len(z) == 0)
    if empty_mask.any():
        df.loc[empty_mask, "tokens"] = df.loc[empty_mask, "text_clean"].map(tokenize)

    # company id (name only)
    df["company_id"] = df["source_file"].map(_derive_company_id)

    # date placeholder if missing
    if "date" not in df.columns:
        df["date"] = pd.NaT

    df["n_tokens"] = df["tokens"].apply(len)
    print(f"[load_reviews] rows={len(df)} companies≈{df['company_id'].nunique()} tokens_mean={df['n_tokens'].mean():.1f}")
    return df

def load_vocab(vpath: Path) -> Dict[str, int]:
    _need(vpath, "tfidf_vocab.json")
    vocab_raw = json.load(open(vpath, "r", encoding="utf-8"))
    vocab_map: Dict[str,int] = {}
    if all(isinstance(k, str) and isinstance(v, int) for k,v in vocab_raw.items()):
        for tok, idx in vocab_raw.items():
            vocab_map[norm_term(tok)] = int(idx)
    else:
        for k, tok in vocab_raw.items():
            vocab_map[norm_term(tok)] = int(k)
    return vocab_map

def load_tfidf(npz_path: Path) -> sparse.csr_matrix:
    _need(npz_path, "tfidf_reviews.npz")
    loader = np.load(npz_path)
    if {"data","indices","indptr","shape"} <= set(loader.files):
        return sparse.csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
    try:
        from scipy.sparse import load_npz
        return load_npz(str(npz_path))
    except Exception as e:
        raise RuntimeError(f"Unrecognized TF-IDF npz format: {e}")

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
            batch_s = [" ".join(s.split()[:60]) for s in batch_s]  # clip very long sentences
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

# ---------------- Phrase/negation helpers ----------------
_SENT_SPLIT = re.compile(r"[.!?]+")
def simple_sent_split(txt: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]

def find_phrase_positions(tokens: List[str], phrase_term: str) -> List[Tuple[int,int]]:
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

def within_window(span_a: Tuple[int,int], span_b: Tuple[int,int], k: int) -> bool:
    aL, aR = span_a; bL, bR = span_b
    return not (aL > bR + k or bL > aR + k)

_SPACY_OK = False
try:
    import spacy
    _nlp = None
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

def detect_negated_spans(tokens: List[str], cfg_negators: List[str]) -> Set[int]:
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

# ---------------- Main ----------------
def main():
    set_seed(42)

    cfg = load_config(CONFIG_JSON)
    reviews = load_reviews(REVIEWS_PATH)
    goals = load_goals(GOALS_PATH)
    vocab  = load_vocab(VOCAB_PATH)
    tfidf  = load_tfidf(TFIDF_NPZ)

    print(f"[load] reviews={len(reviews)} companies≈{reviews['company_id'].nunique()} terms={len(vocab)} tfidf={tfidf.shape}")

    # flatten goal terms → targets
    term2targets: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
    guardrails = {g: [norm_term(x) for x in goals[g]["guardrails"]] for g in goals}
    for g in goals:
        for t in goals[g]["fulfillment"]:
            term2targets[t].append((g, "F"))
        for t in goals[g]["hindrance"]:
            term2targets[t].append((g, "H"))

    # vocab columns for terms we have
    term2col: Dict[str, int] = {t: vocab[t] for t in term2targets.keys() if t in vocab}

    # scoring knobs
    T_W_UNI, T_W_BI, CAP = 0.7, 1.0, 3
    GAMMA, DELTA = 0.8, 0.3
    GUARD_WIN = 6
    NEG_FALLBACK_WIN = 5

    # sentiment
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

    # goal scoring
    goal_keys = list(goals.keys())
    outF = {g: [] for g in goal_keys}
    outH = {g: [] for g in goal_keys}
    outG = {g: [] for g in goal_keys}
    outW = {g: [] for g in goal_keys}
    outGw= {g: [] for g in goal_keys}
    negators_cfg = [t.lower() for t in cfg.get("negators", [])]

    for i, row in tqdm(list(reviews.iterrows()), desc="[goals] reviews"):
        toks: List[str] = row["tokens"]
        n = int(row["n_tokens"])

        neg_idx = detect_negated_spans(toks, negators_cfg)

        guard_spans = []
        for g in goal_keys:
            for gr in guardrails[g]:
                guard_spans.extend(find_phrase_positions(toks, gr))

        spans_cache: Dict[str, List[Tuple[int,int]]] = {}
        Fg = {g: 0.0 for g in goal_keys}
        Hg = {g: 0.0 for g in goal_keys}

        for term, targets in term2targets.items():
            spans = spans_cache.get(term)
            if spans is None:
                spans = find_phrase_positions(toks, term)
                spans_cache[term] = spans
            if not spans:
                continue

            # windowed guardrail suppression
            spans_kept = []
            for sp in spans:
                if any(within_window(sp, grsp, GUARD_WIN) for grsp in guard_spans):
                    continue
                spans_kept.append(sp)
            if not spans_kept:
                continue

            col = term2col.get(term, None)
            base_val = 0.0
            if col is not None:
                base_val = float(tfidf[i, col])
            if col is None or base_val == 0.0:
                base_val = float(len(spans_kept))

            wt_ng = T_W_BI if "_" in term else T_W_UNI
            val = base_val * wt_ng

            for g, kind in targets:
                neg_occ = 0; pos_occ = 0
                for (a,b) in spans_kept:
                    if any(k in neg_idx for k in range(a,b)):
                        neg_occ += 1
                    else:
                        pos_occ += 1
                if pos_occ + neg_occ == 0:
                    continue
                v_pos = val * (pos_occ / (pos_occ + neg_occ))
                v_neg = val * (neg_occ / (pos_occ + neg_occ))
                if kind == "F":
                    Fg[g] += v_pos
                    Hg[g] += GAMMA * v_neg
                else:
                    Hg[g] += v_pos
                    Fg[g] += DELTA * v_neg

        Ln = 1.0 + math.log(1 + n)
        for g in goal_keys:
            Fn = Fg[g] / Ln
            Hn = Hg[g] / Ln
            G  = Fn - Hn

            Srev = float(row["S_raw"])
            sgn = 1 if G > 0 else -1 if G < 0 else 0
            eps, tau, lo, hi = 0.08, 0.50, 0.3, 1.2  # narrower neutral band
            w = 0.5 + 0.5 * sgn * math.tanh(max(abs(Srev) - eps, 0.0) / tau)
            w = float(np.clip(w, lo, hi))
            Gw = w * G

            outF[g].append(Fn); outH[g].append(Hn); outG[g].append(G)
            outW[g].append(w);  outGw[g].append(Gw)

    # ---------------- Outputs ----------------
    rev_out = pd.DataFrame({
        "review_id": reviews["review_id"],
        "company_id": reviews["company_id"],
        "date": reviews["date"],
        "n_tokens": reviews["n_tokens"],
        "S_raw": reviews["S_raw"]
    })
    for g in goal_keys:
        rev_out[f"F_norm_{g}"] = outF[g]
        rev_out[f"H_norm_{g}"] = outH[g]
        rev_out[f"G_raw_{g}"]  = outG[g]
        rev_out[f"w_sent_{g}"] = outW[g]
        rev_out[f"G_weighted_{g}"] = outGw[g]

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
        gmeans = grp[[f"G_raw_{g}", f"G_weighted_{g}"]].mean().reset_index()
        gmeans = gmeans.rename(columns={f"G_raw_{g}":f"G_mean_raw_{g}", f"G_weighted_{g}":f"G_mean_weighted_{g}"})
        comp = comp.merge(gmeans, on="company_id")
        muG = float(comp[f"G_mean_weighted_{g}"].mean())
        comp[f"G_smoothed_weighted_{g}"] = eb(comp[f"G_mean_weighted_{g}"],
                                              comp["n_reviews"].astype(float), lamG, muG)

    # ----- Write outputs (CSV-only) -----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rev_path_csv  = OUT_DIR / "review_scores.csv"
    cmp_path_csv  = OUT_DIR / "company_scores.csv"
    per_dir       = OUT_DIR / "per_company"
    per_dir.mkdir(exist_ok=True)

    # 1) Global review-level scores (all reviews)
    rev_out.to_csv(rev_path_csv, index=False)

    # 2) Company-level aggregates
    comp.to_csv(cmp_path_csv, index=False)

    # 3) Per-company review-level (trimmed)
    import re as _re
    g_cols = [c for c in rev_out.columns if c.startswith("G_raw_") or c.startswith("G_weighted_")]
    base_cols = ["review_id","company_id","S_raw"]
    keep_cols = base_cols + g_cols

    written_per_company = []
    for cid, sub in rev_out.groupby("company_id", dropna=False):
        safe = _re.sub(r"[^A-Za-z0-9_]+", "_", str(cid)).strip("_") or "unknown"
        (per_dir / f"{safe}.csv").write_text(sub[keep_cols].to_csv(index=False))
        written_per_company.append(str(per_dir / f"{safe}.csv"))

    # Run report JSON
    run_meta = {
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "batch": best_batch(),
        "max_length": 256,
        "eps": 0.08, "tau": 0.50,
        "gamma": 0.8, "delta": 0.3,
        "t_w_uni": 0.7, "t_w_bi": 1.0, "cap": 3,
        "eb_lambda_S": 80.0, "eb_lambda_G": 80.0,
        "guard_window": 6,
        "neg_fallback_window": 5,
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
        open(OUT_DIR / "run_report.json", "w"),
        indent=2
    )

    print(f"[done] CSVs:\n - {rev_path_csv}\n - {cmp_path_csv}\n - per-company (trimmed) in {per_dir}")

if __name__ == "__main__":
    main()
