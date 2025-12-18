import re, os, json, glob, html, argparse, unicodedata
from pathlib import Path
from typing import List, Iterable, Dict, Any
import numpy as np
import pandas as pd
from unidecode import unidecode
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
SPACY_STOP = nlp.Defaults.stop_words

# ----------------------------- Config knobs -----------------------------
DEFAULT_STOP_EXTRA = {
    "company","companies","manager","managers","coworker","coworkers","employee","employees",
    "workplace","work","worked","working","year","years","month","months",
    "pros","cons","review","reviews","glassdoor","tldr"
}

NEGATORS = {"no","not","never","without","hardly","rarely","scarcely","barely","seldom"}
NEG_WINDOW = 3

MIN_TOKEN_LEN = 2
MIN_DOC_LEN = 3
NGRAM_RANGE = (1,2)
MAX_FEATURES = 60000
MIN_DF = 5
MAX_DF = 0.9

ID_COL_CANDIDATES = ["review_id","id","gid"]
TEXT_COL_CANDIDATES = ["text","body","review_text","content"]
TITLE_COL = "title"
PROS_COL  = "pros"
CONS_COL  = "cons"

# --------------------------- Normalization utils -------------------------
URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
WS_RE    = re.compile(r'\s+')

def normalize_text(s: str) -> str:
    """HTML/Unicode cleanup, lowercase, keep digits as-is, mask only URLs/emails."""
    if not isinstance(s, str) or not s.strip():
        return ""
    s = html.unescape(s)
    s = s.replace("\u00A0", " ")
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub(" <url> ", s)
    s = EMAIL_RE.sub(" <email> ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9<>\-']+", " ", s)
    s = WS_RE.sub(" ", s).strip()
    s = unidecode(s)
    return s

def spacy_tokenize_lemma(s: str) -> List[str]:
    """Tokenize + lemmatize; keep numbers; drop stopwords and domain filler."""
    if not s:
        return []
    doc = nlp(s)

    toks = []
    for t in doc:
        if t.like_email or t.text == "<email>":
            toks.append("<email>")
            continue
        if t.text == "<url>":
            toks.append("<url>")
            continue

        if t.like_num:
            numtok = t.text.strip()
            if len(numtok) >= MIN_TOKEN_LEN:
                toks.append(numtok)
            continue

        lemma = t.lemma_.strip().lower()
        if not lemma or lemma == "-pron-":
            lemma = t.text.lower()

        if lemma in SPACY_STOP or lemma in DEFAULT_STOP_EXTRA:
            continue
        if len(lemma) < MIN_TOKEN_LEN:
            continue
        if not any(ch.isalpha() for ch in lemma):
            continue

        toks.append(lemma)
    return toks

def apply_negation(tokens: List[str], window: int = NEG_WINDOW) -> List[str]:
    """Append _NEG to the next `window` content tokens after a negator."""
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        if tok in NEGATORS:
            k = 0
            j = i + 1
            while j < len(tokens) and k < window:
                nxt = tokens[j]
                if nxt not in {"<url>","<email>"} and nxt not in NEGATORS:
                    out.append(f"{nxt}_NEG")
                    j += 1
                    k += 1
                else:
                    out.append(nxt)
                    j += 1
            i = j
        else:
            i += 1
    return out

def join_cols(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        if c in row and isinstance(row[c], str) and row[c].strip():
            parts.append(row[c])
    return " \n ".join(parts)

# ------------------------------- IO utils --------------------------------
def detect_id_col(df: pd.DataFrame) -> str | None:
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def build_text_raw(df: pd.DataFrame) -> pd.Series:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            base = df[c].fillna("")
            break
    else:
        base = pd.Series([""]*len(df))
    cols = [TITLE_COL, PROS_COL, CONS_COL]
    extra = df.apply(lambda r: join_cols(r, cols), axis=1)
    return (base.fillna("") + "\n" + extra.fillna("")).str.strip()

# ----------------------------- Vectorization -----------------------------
def vectorize_tfidf(pretokenized, max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_df=MAX_DF):
    docs = [" ".join(toks) for toks in pretokenized]
    vec = TfidfVectorizer(
        analyzer="word",
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
        norm="l2",
        sublinear_tf=True
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out().tolist()
    return X, vocab


# ------------------------------- Pipeline --------------------------------
def process_files(pattern: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df["source_file"] = Path(fp).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    id_col = detect_id_col(df)
    if id_col is None:
        df["review_id"] = np.arange(len(df), dtype=int)
        id_col = "review_id"

    df["text_raw"] = build_text_raw(df).astype(str)

    df["text_norm"] = df["text_raw"].map(normalize_text)
    df["tokens"] = df["text_norm"].map(spacy_tokenize_lemma)

    df["tokens"] = df["tokens"].map(apply_negation)
    df["tokens"] = df["tokens"].map(lambda ts: [t for t in ts if t not in NEGATORS])

    lens = df["tokens"].map(len)
    df = df[lens >= MIN_DOC_LEN].reset_index(drop=True)

    X, vocab = vectorize_tfidf(df["tokens"].tolist())

    df_out = df[[id_col, "source_file", "text_raw", "text_norm", "tokens"]].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_dir / "combined_reviews.parquet", index=False)

    sparse.save_npz(out_dir / "tfidf_reviews.npz", X)
    with open(out_dir / "tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump({i: term for i, term in enumerate(vocab)}, f, ensure_ascii=False, indent=2)
    df_out[[id_col, "source_file"]].to_csv(out_dir / "doc_index.csv", index=False)

    config = dict(
        stop_extra=sorted(list(DEFAULT_STOP_EXTRA)),
        negators=sorted(list(NEGATORS)),
        neg_window=NEG_WINDOW,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES,
        min_doc_len=MIN_DOC_LEN,
        files=files,
        rows=df_out.shape[0],
        cols=len(vocab),
        id_col=id_col,
        keep_numbers=True
    )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[done] docs={config['rows']} features={config['cols']}")
    print(f"Saved → {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="pattern", required=True,
                    help="Glob for cleaned CSVs, e.g., C:/.../cleaned/*.csv")
    ap.add_argument("--out", dest="out_dir", required=True,
                    help="Output directory for features")
    args = ap.parse_args()
    process_files(args.pattern, Path(args.out_dir))

if __name__ == "__main__":
    main()
