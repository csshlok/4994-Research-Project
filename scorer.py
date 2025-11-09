from __future__ import annotations
import os, json, math, re, warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re, ast


ROOT = Path(__file__).resolve().parent

FEATURES_DIR = ROOT / "features_exctract"
CONFIG_DIR   = ROOT / "config"
OUT_DIR      = ROOT / "out"

REVIEWS_PATH = FEATURES_DIR / "combined_reviews.parquet"
TFIDF_PATH   = FEATURES_DIR / "tfidf_reviews.npz"
VOCAB_PATH   = FEATURES_DIR / "tfidf_vocab.json"
GOALS_PATH   = CONFIG_DIR / "goal_dict.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[auto] Using files:\n"
      f"  reviews: {REVIEWS_PATH}\n"
      f"  tfidf:   {TFIDF_PATH}\n"
      f"  vocab:   {VOCAB_PATH}\n"
      f"  goals:   {GOALS_PATH}\n"
      f"  outdir:  {OUT_DIR}\n", flush=True)


def load_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    text_candidates = [
        "text_clean", "clean_text", "review_text", "review", "text",
        "body", "content", "full_text", "comments", "summary"
    ]
    text_col = next((c for c in text_candidates if c in df.columns), None)

    if text_col is None:
        if {"pros", "cons"}.issubset(df.columns):
            df["__joined_text__"] = (
                df["pros"].fillna("").astype(str).str.strip() + ". " +
                df["cons"].fillna("").astype(str).str.strip()
            )
            text_col = "__joined_text__"
        else:
            raise ValueError(
                f"No text column found in {path}. "
                f"Available columns: {list(df.columns)}. "
                f"Tried {text_candidates} or (pros+cons)."
            )

    if "review_id" not in df.columns:
        for alt in ["id", "doc_id", "reviewID"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "review_id"})
                break
    if "review_id" not in df.columns:
        raise ValueError("Missing 'review_id' column (or fallback id/doc_id).")

    if "company_id" not in df.columns:
        for alt in ["company", "employer", "firm_id", "companyName", "companyID"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "company_id"})
                break
    if "company_id" not in df.columns:
        raise ValueError("Missing 'company_id' column (or fallback company/employer/firm_id).")

    date_alt = next((c for c in ["date", "review_date", "timestamp", "created_at"] if c in df.columns), None)
    df["date"] = pd.to_datetime(df[date_alt], errors="coerce") if date_alt else pd.NaT

    tok_alt = next((c for c in ["tokens", "token_list", "lemmas"] if c in df.columns), None)

    def to_tokens_from_text(s: str) -> list[str]:
        s = "" if s is None else str(s)
        return re.findall(r"[A-Za-z]+", s.lower())

    if tok_alt is None:
        df["tokens"] = df[text_col].fillna("").map(to_tokens_from_text)
    else:
        def normalize_tok(x):
            if isinstance(x, list):
                return [str(t).lower() for t in x]
            if isinstance(x, str):
                s = x.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        arr = ast.literal_eval(s)
                        return [str(t).lower() for t in arr]
                    except Exception:
                        pass
                return [t for t in s.lower().split() if any(ch.isalpha() for ch in t)]
            return []
        df["tokens"] = df[tok_alt].apply(normalize_tok)

    df["n_tokens"] = df["tokens"].apply(len)

    df["text_clean"] = df[text_col].fillna("").astype(str)


    return df

def load_vocab(vpath: Path) -> Dict[str,int]:
    vocab = json.load(open(vpath,"r",encoding="utf-8"))
    if all(isinstance(v,int) for v in vocab.values()):
        return vocab
    if all(k.isdigit() for k in vocab.keys()):
        return {v:int(k) for k,v in vocab.items()}
    raise ValueError("Unrecognized vocab format.")

def load_goals(gpath: Path) -> Dict[str,Dict[str,List[str]]]:
    raw = json.load(open(gpath,"r",encoding="utf-8"))
    mapping = {
        "physiological":"phys","self_protection":"selfprot","affiliation":"aff",
        "status_esteem":"stat","mate_acquisition":"m_acq","mate_retention":"m_ret","family_care":"fam"}
    goals = {}
    for k,v in raw.items():
        kk = mapping.get(k,k)
        goals[kk] = {p:[t.lower() for t in v.get(p,[])] for p in ["fulfillment","hindrance","guardrails"]}
    return goals



class SentimentModel:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.tok=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        id2=self.model.config.id2label
        lid={v.lower():k for k,v in id2.items()}
        self.i_neg, self.i_neu, self.i_pos = lid.get("negative",0), lid.get("neutral",1), lid.get("positive",2)
    @torch.no_grad()
    def score(self,sents,batch=64):
        out=[]
        for i in range(0,len(sents),batch):
            batch_s=sents[i:i+batch]
            toks=self.tok(batch_s,padding=True,truncation=True,max_length=256,return_tensors="pt").to(self.device)
            p=torch.softmax(self.model(**toks).logits,dim=-1).cpu().numpy()
            p=p[:,[self.i_neg,self.i_neu,self.i_pos]]
            out.append(p)
        return np.vstack(out) if out else np.zeros((0,3))
    @staticmethod
    def probs_to_scalar(pn,p0,pp):
        val=(pp-pn)/(1-p0/2+1e-6)
        return float(np.clip(val,-1,1))


def simple_sent_split(txt:str)->List[str]:
    return [s.strip() for s in re.split(r"[.!?]+",txt) if s.strip()]

def collect_counts(tokens:List[str])->Counter:
    bg=[f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    c=Counter(tokens); c.update(bg); return c

def optionB_weight(G,S,eps=0.15,tau=0.75,lo=0.3,hi=1.2):
    if G==0: return 1.0,0.0
    sgn=1 if G>0 else -1
    w=0.5+0.5*sgn*math.tanh(max(abs(S)-eps,0)/tau)
    w=float(np.clip(w,lo,hi))
    return w,G*w



def main():
    reviews=load_reviews(REVIEWS_PATH)
    X=load_npz(TFIDF_PATH)
    vocab=load_vocab(VOCAB_PATH)
    goals=load_goals(GOALS_PATH)
    goal_keys=list(goals.keys())

    print(f"[load] {len(reviews)} reviews, {X.shape[1]} vocab terms")


    sm=SentimentModel()
    all_sents=[]; ridx=[]; slen=[]
    for i,txt in enumerate(reviews["text_clean"].fillna("")):
        sents=simple_sent_split(txt) or [txt]
        for s in sents:
            all_sents.append(s)
            ridx.append(i)
            slen.append(max(1,len(s.split())))
    print(f"[sent] scoring {len(all_sents)} sentences…")
    probs=sm.score(all_sents,64)
    s_scalar=[sm.probs_to_scalar(p[0],p[1],p[2]) for p in probs]


    S=np.zeros(len(reviews))
    d=defaultdict(list); w=defaultdict(list)
    for s,r,l in zip(s_scalar,ridx,slen):
        d[r].append(s); w[r].append(l)
    for i in range(len(reviews)):
        S[i]=np.average(d[i],weights=w[i]) if d[i] else 0
    reviews["S_raw"]=S


    print("[goals] computing fulfillment/hindrance…")
    def ensure_vars(lst): 
        return list(set(lst+[t.replace(" ","_") for t in lst if " " in t]))
    goal_terms={g:{"F":ensure_vars(v["fulfillment"]),"H":ensure_vars(v["hindrance"])} for g,v in goals.items()}
    guardrails={g:set(v["guardrails"]) for g,v in goals.items()}
    gamma,delta=0.8,0.3
    t_w_uni,t_w_bi=0.7,1.0
    cap=3

    outF={g:[] for g in goal_keys}; outH={g:[] for g in goal_keys}
    outG={g:[] for g in goal_keys}; outW={g:[] for g in goal_keys}; outGw={g:[] for g in goal_keys}

    for i,row in reviews.iterrows():
        toks=row["tokens"]; text=row["text_clean"].lower(); n=len(toks)
        counts=collect_counts(toks)
        neg_idx=set()
        for j,t in enumerate(toks):
            if t in {"no","not","never","without","hardly","scarcely","lack"}:
                neg_idx.update(range(j+1,min(j+6,len(toks))))
        def negated(term):
            for p in term.split("_"):
                for j,t in enumerate(toks):
                    if t==p and j in neg_idx: return True
            return False
        for g in goal_keys:
            F=H=0.0
            for t in goal_terms[g]["F"]:
                if any(gr in text for gr in guardrails[g]): continue
                f=counts.get(t,0)
                if f==0: continue
                tf=1+math.log(min(f,cap))
                wt=t_w_bi if "_" in t else t_w_uni
                if not negated(t): F+=wt*tf
                else: H+=gamma*wt*tf
            for t in goal_terms[g]["H"]:
                if any(gr in text for gr in guardrails[g]): continue
                f=counts.get(t,0)
                if f==0: continue
                tf=1+math.log(min(f,cap))
                wt=t_w_bi if "_" in t else t_w_uni
                if not negated(t): H+=wt*tf
                else: F+=delta*wt*tf
            Fn=F/(1+math.log(1+n)); Hn=H/(1+math.log(1+n))
            G=Fn-Hn
            w,Gw=optionB_weight(G,row["S_raw"])
            outF[g].append(Fn); outH[g].append(Hn)
            outG[g].append(G); outW[g].append(w); outGw[g].append(Gw)


    rev_out=pd.DataFrame({
        "review_id":reviews["review_id"],
        "company_id":reviews["company_id"],
        "date":reviews["date"],
        "n_tokens":reviews["n_tokens"],
        "S_raw":reviews["S_raw"]
    })
    for g in goal_keys:
        rev_out[f"F_norm_{g}"]=outF[g]
        rev_out[f"H_norm_{g}"]=outH[g]
        rev_out[f"G_raw_{g}"]=outG[g]
        rev_out[f"w_sent_{g}"]=outW[g]
        rev_out[f"G_weighted_{g}"]=outGw[g]


    print("[aggregate] company scores…")
    grp=rev_out.groupby("company_id",dropna=False)
    comp=grp["S_raw"].agg(["count","mean","std"]).rename(columns={"count":"n_reviews","mean":"S_mean","std":"S_std"}).reset_index()
    lamS=80; lamG=80
    muS=comp["S_mean"].mean()
    eb=lambda m,n,l,mu:(n/(n+l))*m+(l/(n+l))*mu
    comp["S_smoothed"]=eb(comp["S_mean"],comp["n_reviews"],lamS,muS)
    comp["SE"]=comp["S_std"]/np.sqrt(comp["n_reviews"])
    comp["S_CI_low"]=comp["S_smoothed"]-1.96*comp["SE"]
    comp["S_CI_high"]=comp["S_smoothed"]+1.96*comp["SE"]
    for g in goal_keys:
        cg=grp[[f"G_raw_{g}",f"G_weighted_{g}"]].mean().reset_index()
        cg=cg.rename(columns={f"G_raw_{g}":f"G_mean_raw_{g}",f"G_weighted_{g}":f"G_mean_weighted_{g}"})
        comp=comp.merge(cg,on="company_id")
        muG=comp[f"G_mean_weighted_{g}"].mean()
        comp[f"G_smoothed_weighted_{g}"]=eb(comp[f"G_mean_weighted_{g}"],comp["n_reviews"],lamG,muG)


    rev_out.to_parquet(OUT_DIR/"review_scores.parquet",index=False)
    comp.to_parquet(OUT_DIR/"company_scores.parquet",index=False)
    json.dump({"goals":goal_keys,"N_reviews":len(reviews)},open(OUT_DIR/"run_report.json","w"),indent=2)
    print(f"[done] Results saved in {OUT_DIR}")



if __name__=="__main__":
    main()
