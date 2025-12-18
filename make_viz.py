from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Paths (edit if needed)
# ----------------------------
DEFAULT_COMPANY_CSV = r"C:/Users/csshl\Desktop/4994-Scrapper/out/company_scores.csv"
DEFAULT_REVIEW_CSV  = r"C:/Users/csshl\Desktop/4994-Scrapper/out/review_scores.csv"
DEFAULT_PER_COMPANY_DIR = r"C:/Users/csshl\Desktop/4994-Scrapper/out/per_company"
DEFAULT_OUT_DIR = r"C:/Users/csshl\Desktop/4994-Scrapper/out/figures"


# ----------------------------
# Goal display names (use these in ALL plots)
# ----------------------------
GOAL_LABELS = {
    "phys": "Physiological Needs",
    "selfprot": "Self Protection",
    "aff": "Affiliation",
    "stat": "Status & Esteem",
    "fam": "Family Care",
}


def goal_display_names(goal_codes: List[str]) -> List[str]:
    return [GOAL_LABELS.get(g, g) for g in goal_codes]


# ----------------------------
# Helpers
# ----------------------------
def _need(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_or_show(fig: plt.Figure, outpath: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def ordered_companies_from_per_company(per_dir: Path) -> List[str]:
    """
    Uses filenames in per_company/ to define a stable company list:
      accenture.csv -> accenture
      boozallen.csv -> boozallen
    """
    if not per_dir.exists():
        return []
    files = sorted([p for p in per_dir.glob("*.csv") if p.is_file()])
    return [p.stem for p in files]


def coerce_company_id(s: str) -> str:
    return str(s).strip().lower()


def get_goal_codes_from_company_scores(comp: pd.DataFrame) -> List[str]:
    """
    Detect goal codes from columns like:
      G_smoothed_final_phys, G_smoothed_final_aff, ...
    """
    cols = list(comp.columns)
    codes = []
    prefix = "G_smoothed_final_"
    for c in cols:
        if c.startswith(prefix):
            codes.append(c.replace(prefix, ""))
    preferred = ["phys", "selfprot", "aff", "stat", "fam"]
    ordered = [g for g in preferred if g in codes] + [g for g in codes if g not in preferred]
    return ordered


def compute_goal_goal_corr(comp: pd.DataFrame, goal_codes: List[str]) -> pd.DataFrame:
    cols = [f"G_smoothed_final_{g}" for g in goal_codes]
    X = comp[cols].copy()
    X.columns = goal_codes
    return X.corr()


def compute_company_goal_matrix(comp: pd.DataFrame, goal_codes: List[str]) -> pd.DataFrame:
    """
    Returns a matrix with index=company_id and columns=goal_codes
    values = G_smoothed_final_{goal}
    """
    mat_cols = [f"G_smoothed_final_{g}" for g in goal_codes]
    M = comp[["company_id"] + mat_cols].copy()
    M["company_id"] = M["company_id"].map(coerce_company_id)
    M = M.set_index("company_id")
    M.columns = goal_codes
    return M


def get_review_goal_cols(rev: pd.DataFrame) -> List[str]:
    return [c for c in rev.columns if c.startswith("G_final_")]


def set_company_order(companies: List[str], comp_df: pd.DataFrame) -> List[str]:
    """
    Prefer per_company file order (the “10 companies”), else fall back to comp_df ordering.
    """
    if companies:
        return companies
    return sorted(comp_df["company_id"].map(coerce_company_id).unique().tolist())


# ----------------------------
# Plot 1: Company sentiment leaderboard
# ----------------------------
def plot_company_sentiment_leaderboard(comp: pd.DataFrame, company_order: List[str],
                                      out_dir: Path, show: bool) -> None:
    col = "S_smoothed" if "S_smoothed" in comp.columns else ("S_mean" if "S_mean" in comp.columns else None)
    if col is None:
        raise ValueError("company_scores.csv missing S_smoothed/S_mean columns.")

    df = comp.copy()
    df["company_id"] = df["company_id"].map(coerce_company_id)

    df = df[df["company_id"].isin(company_order)].copy()
    df = df.set_index("company_id").loc[company_order].reset_index()

    df = df.sort_values(col, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df["company_id"], df[col])
    ax.invert_yaxis()
    ax.set_title("Company Sentiment Leaderboard (Smoothed)")
    ax.set_xlabel("Sentiment (S_smoothed)")
    ax.set_ylabel("Company")
    ax.axvline(0, linewidth=1)

    outpath = out_dir / "01_company_sentiment_leaderboard.png"
    save_or_show(fig, outpath, show)


# ----------------------------
# Plot 2: Company × goal heatmap
# ----------------------------
def plot_company_goal_heatmap(comp: pd.DataFrame, company_order: List[str], goal_codes: List[str],
                              out_dir: Path, show: bool) -> None:
    M = compute_company_goal_matrix(comp, goal_codes)
    M = M.loc[[c for c in company_order if c in M.index]]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(M.values, aspect="auto")

    ax.set_title("Company × Goal Heatmap (Smoothed Final Goal Scores)")
    ax.set_xticks(np.arange(len(goal_codes)))
    ax.set_xticklabels(goal_display_names(goal_codes))
    ax.set_yticks(np.arange(len(M.index)))
    ax.set_yticklabels(M.index.tolist())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Goal score (smoothed final)")

    outpath = out_dir / "02_company_goal_heatmap.png"
    save_or_show(fig, outpath, show)


# ----------------------------
# Plot 3: Goal–goal correlation heatmap
# ----------------------------
def plot_goal_goal_corr(comp: pd.DataFrame, goal_codes: List[str], out_dir: Path, show: bool) -> None:
    corr = compute_goal_goal_corr(comp, goal_codes)
    display = goal_display_names(goal_codes)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(corr.values, aspect="equal")

    ax.set_title("Goal–Goal Correlation (Company Level)")
    ax.set_xticks(np.arange(len(goal_codes)))
    ax.set_xticklabels(display)
    ax.set_yticks(np.arange(len(goal_codes)))
    ax.set_yticklabels(display)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    outpath = out_dir / "03_goal_goal_correlation_heatmap.png"
    save_or_show(fig, outpath, show)


# ----------------------------
# Plot 4: Goal fulfillment vs hindrance per company (stacked bars)
# ----------------------------
def plot_fulfillment_vs_hindrance(companies: List[str], review: pd.DataFrame,
                                 out_dir: Path, show: bool) -> None:
    f_cols = [c for c in review.columns if c.startswith("F_raw_")]
    h_cols = [c for c in review.columns if c.startswith("H_raw_")]
    if not f_cols or not h_cols:
        raise ValueError("review_scores.csv missing F_raw_* and/or H_raw_* columns.")

    goal_codes = [c.replace("F_raw_", "") for c in f_cols]
    goal_codes = [g for g in ["phys", "selfprot", "aff", "stat", "fam"] if g in goal_codes] + \
                 [g for g in goal_codes if g not in ["phys", "selfprot", "aff", "stat", "fam"]]

    df = review.copy()
    df["company_id"] = df["company_id"].map(coerce_company_id)
    df = df[df["company_id"].isin(companies)].copy()

    n_goals = len(goal_codes)
    fig_h = 2.2 * n_goals + 1.0
    fig, axes = plt.subplots(n_goals, 1, figsize=(11, fig_h), sharex=True)
    if n_goals == 1:
        axes = [axes]

    for ax, g in zip(axes, goal_codes):
        f = df.groupby("company_id")[f"F_raw_{g}"].mean().reindex(companies).fillna(0.0)
        h = df.groupby("company_id")[f"H_raw_{g}"].mean().reindex(companies).fillna(0.0)

        ax.bar(companies, f.values, label="Fulfillment (mean)")
        ax.bar(companies, -h.values, label="Hindrance (mean)")
        ax.axhline(0, linewidth=1)

        ax.set_title(f"{GOAL_LABELS.get(g, g)} — Mean Fulfillment vs Hindrance")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Goal Fulfillment vs Hindrance per Company (Review-Level Means)", y=0.99)

    outpath = out_dir / "04_fulfillment_vs_hindrance_stacked.png"
    save_or_show(fig, outpath, show)


# ----------------------------
# Plot 5: Zero-goal vs goal-hit sentiment comparison
# ----------------------------
def plot_zero_goal_vs_goalhit_sentiment(review: pd.DataFrame, out_dir: Path, show: bool) -> None:
    goal_cols = get_review_goal_cols(review)
    if not goal_cols:
        raise ValueError("review_scores.csv missing G_final_* columns.")

    df = review.copy()
    hit = (df[goal_cols].abs().sum(axis=1) > 0)
    s0 = df.loc[~hit, "S_raw"].dropna().to_numpy()
    s1 = df.loc[hit, "S_raw"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([s0, s1], labels=["0 goals hit", "≥1 goal hit"], showmeans=True)
    ax.set_title("Sentiment: Zero-goal vs Goal-hit Reviews")
    ax.set_ylabel("Sentiment (S_raw)")
    ax.axhline(0, linewidth=1)

    outpath = out_dir / "05_zero_goal_vs_goalhit_sentiment.png"
    save_or_show(fig, outpath, show)


# ----------------------------
# Plot 6: Radar charts for all companies
# ----------------------------
def radar_chart(ax, labels: List[str], values_a: np.ndarray, values_b: np.ndarray,
                title: str, label_a: str = "Fulfillment", label_b: str = "Hindrance") -> None:
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    angles = np.concatenate([angles, angles[:1]])
    va = np.concatenate([values_a, values_a[:1]])
    vb = np.concatenate([values_b, values_b[:1]])

    ax.plot(angles, va, linewidth=2, label=label_a)
    ax.fill(angles, va, alpha=0.15)

    ax.plot(angles, vb, linewidth=2, label=label_b)
    ax.fill(angles, vb, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=15)
    ax.set_ylim(0, max(1e-6, float(max(va.max(), vb.max()))) * 1.1)


def plot_company_radar_all(companies: List[str], review: pd.DataFrame, out_dir: Path, show: bool) -> None:
    f_cols = [c for c in review.columns if c.startswith("F_raw_")]
    h_cols = [c for c in review.columns if c.startswith("H_raw_")]
    if not f_cols or not h_cols:
        raise ValueError("review_scores.csv missing F_raw_* and/or H_raw_* columns.")

    goal_codes = [c.replace("F_raw_", "") for c in f_cols]
    goal_codes = [g for g in ["phys", "selfprot", "aff", "stat", "fam"] if g in goal_codes] + \
                 [g for g in goal_codes if g not in ["phys", "selfprot", "aff", "stat", "fam"]]

    labels = goal_display_names(goal_codes)

    df = review.copy()
    df["company_id"] = df["company_id"].map(coerce_company_id)

    n = len(companies)
    rows = 2
    cols = int(np.ceil(n / rows))
    fig = plt.figure(figsize=(3.2 * cols, 3.2 * rows))

    for i, cid in enumerate(companies):
        sub = df[df["company_id"] == cid]

        Fvals = np.array([sub[f"F_raw_{g}"].mean() if len(sub) else 0.0 for g in goal_codes], dtype=float)
        Hvals = np.array([sub[f"H_raw_{g}"].mean() if len(sub) else 0.0 for g in goal_codes], dtype=float)

        ax = fig.add_subplot(rows, cols, i + 1, polar=True)
        radar_chart(
            ax=ax,
            labels=labels,
            values_a=Fvals,
            values_b=Hvals,
            title=cid,
            label_a="Fulfillment",
            label_b="Hindrance"
        )

    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right")
    fig.suptitle("Company Radar Charts: Fulfillment vs Hindrance (Review-Level Means)", y=0.98)

    outpath = out_dir / "06_company_radar_all.png"
    save_or_show(fig, outpath, show)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Glassdoor goal/sentiment visualizations (matplotlib).")
    parser.add_argument("--company_csv", default=DEFAULT_COMPANY_CSV)
    parser.add_argument("--review_csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--per_company_dir", default=DEFAULT_PER_COMPANY_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--show", action="store_true", help="Show plots interactively (also saves).")
    args = parser.parse_args()

    company_csv = Path(args.company_csv)
    review_csv = Path(args.review_csv)
    per_dir = Path(args.per_company_dir)
    out_dir = Path(args.out_dir)

    _need(company_csv, "company_scores.csv")
    _need(review_csv, "review_scores.csv")
    ensure_outdir(out_dir)

    comp = pd.read_csv(company_csv)
    rev = pd.read_csv(review_csv)

    companies = ordered_companies_from_per_company(per_dir)
    companies = [coerce_company_id(x) for x in companies]
    companies = set_company_order(companies, comp)

    goal_codes = get_goal_codes_from_company_scores(comp)

    print("[viz] companies:", companies)
    print("[viz] goals:", goal_display_names(goal_codes))
    print("[viz] saving to:", out_dir)

    plot_company_sentiment_leaderboard(comp, companies, out_dir, args.show)
    plot_company_goal_heatmap(comp, companies, goal_codes, out_dir, args.show)
    plot_goal_goal_corr(comp, goal_codes, out_dir, args.show)
    plot_fulfillment_vs_hindrance(companies, rev, out_dir, args.show)
    plot_zero_goal_vs_goalhit_sentiment(rev, out_dir, args.show)
    plot_company_radar_all(companies, rev, out_dir, args.show)

    print("\n[viz] done. Files written:")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
