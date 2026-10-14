"""
STEP 5b — Results, Tables & Graphs (Smart Judge, Pure Python)
-------------------------------------------------------------
Reads per-vehicle judge JSON files from:
  <PROJECT_ROOT>/scores/<MODEL_NAME>/*.json

Produces:
  - results/<MODEL_NAME>_judge_aggregates.csv
  - graphs/<MODEL_NAME>_01_means.png
  - graphs/<MODEL_NAME>_02_boxplot_scores.png
  - graphs/<MODEL_NAME>_03_score_distributions.png
  - graphs/<MODEL_NAME>_04_parts_prf_scatter.png          (useful extra #1)
  - graphs/<MODEL_NAME>_05_cost_error_vs_parts_recall.png (useful extra #2)

Plots included:
  1) mean bar chart (scores + key metrics)
  2) boxplot of the 1–5 scores
  3) score distributions (histograms for each score)
  4) scatter: Part precision vs recall (colored by F1)
  5) scatter: Cost abs error vs Part recall (spot systematic cost issues)

Usage:
  pip install pandas matplotlib seaborn
  python scripts/05b_results_smartjudge.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── CONFIG ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\Hunor\EGYETEM\A3.felev\Allamvizsga+tesztek\AutokReportok")
MODEL_NAME   = "summaries"

SCORES_DIR   = PROJECT_ROOT / "scores" / MODEL_NAME
OUT_RESULTS  = PROJECT_ROOT / "results"
OUT_GRAPHS   = PROJECT_ROOT / "graphs" / "test"

OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_GRAPHS.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────


SCORE_COLS = ["score_parts", "score_severity", "score_cost", "score_overall"]
METRIC_COLS = [
    "part_precision", "part_recall", "part_f1",
    "severity_match_rate",
    "cost_abs_error",
    "gt_estimated_repair_cost", "ai_estimated_repair_cost",
]

ID_COLS = ["vehicle_id", "gt_file", "ai_file"]


def load_all_judge_json(scores_dir: Path) -> pd.DataFrame:
    if not scores_dir.exists():
        raise FileNotFoundError(f"Missing scores dir: {scores_dir}")

    rows = []
    bad = 0

    for p in sorted(scores_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_judge_file"] = str(p)
            rows.append(data)
        except Exception:
            bad += 1

    if not rows:
        raise RuntimeError(f"No judge JSON files found in: {scores_dir}")

    if bad:
        print(f"[WARN] {bad} judge files could not be parsed (skipped).")

    df = pd.DataFrame(rows)

    # Ensure numeric columns
    for c in SCORE_COLS + METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def save_aggregates(df: pd.DataFrame) -> Path:
    cols = [c for c in (ID_COLS + SCORE_COLS + METRIC_COLS) if c in df.columns]
    out = df[cols].copy()
    out_path = OUT_RESULTS / f"{MODEL_NAME}_judge_aggregates.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} vehicles)")
    return out_path


def plot_means(df: pd.DataFrame):
    # show score means (1–5)
    means = df[SCORE_COLS].mean(numeric_only=True).sort_values(ascending=False)

    plt.figure(figsize=(8, 4.5))
    ax = means.plot(kind="bar", color="#4EA8E3")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 5.8)
    plt.title(f"Average Smart-Judge Scores ({MODEL_NAME})")
    plt.ylabel("Average score (1–5)")
    plt.xlabel("")
    plt.tight_layout()

    out = OUT_GRAPHS / f"{MODEL_NAME}_01_means.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved: {out}")


def plot_boxplot_scores(df: pd.DataFrame):
    long_df = df.melt(value_vars=[c for c in SCORE_COLS if c in df.columns],
                      var_name="score_type", value_name="score").dropna()

    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=long_df, x="score_type", y="score")
    plt.ylim(0.5, 5.5)
    plt.title(f"Smart-Judge Score Boxplots ({MODEL_NAME})")
    plt.xlabel("")
    plt.ylabel("Score (1–5)")
    plt.tight_layout()

    out = OUT_GRAPHS / f"{MODEL_NAME}_02_boxplot_scores.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved: {out}")


def plot_score_distributions(df: pd.DataFrame):
    # 2x2 grid of histograms for the 4 scores
    cols = [c for c in SCORE_COLS if c in df.columns]

    n = len(cols)
    if n == 0:
        return

    # fixed layout for up to 4
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for i, col in enumerate(cols[:4]):
        ax = axes[i]
        sns.histplot(df[col].dropna(), bins=[1, 2, 3, 4, 5, 6], discrete=True,
                     color="#4EA8E3", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Score (1–5)")
        ax.set_ylabel("Count")
        ax.set_xlim(0.5, 5.5)

    # hide unused
    for j in range(i + 1, 4):
        axes[j].axis("off")

    plt.suptitle(f"Score Distributions ({MODEL_NAME})")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = OUT_GRAPHS / f"{MODEL_NAME}_03_score_distributions.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved: {out}")


# ── Extra plot #1: precision vs recall (colored by F1) ──────────────────────
def plot_parts_prf_scatter(df: pd.DataFrame):
    need = {"part_precision", "part_recall", "part_f1"}
    if not need.issubset(df.columns):
        return

    d = df.dropna(subset=["part_precision", "part_recall", "part_f1"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(6.5, 5.5))
    sc = plt.scatter(d["part_precision"], d["part_recall"],
                     c=d["part_f1"], cmap="viridis", alpha=0.75, s=18)
    plt.colorbar(sc, label="Part F1")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel("Part Precision")
    plt.ylabel("Part Recall")
    plt.title(f"Parts Precision vs Recall ({MODEL_NAME})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    out = OUT_GRAPHS / f"{MODEL_NAME}_04_parts_prf_scatter.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved: {out}")


# ── Extra plot #2: cost error vs part recall ────────────────────────────────
def plot_cost_error_vs_parts_recall(df: pd.DataFrame):
    if "cost_abs_error" not in df.columns or "part_recall" not in df.columns:
        return

    d = df.dropna(subset=["cost_abs_error", "part_recall"]).copy()
    if d.empty:
        return

    # optionally cap extreme errors so plot is readable
    cap = d["cost_abs_error"].quantile(0.98)
    d["cost_abs_error_capped"] = d["cost_abs_error"].clip(upper=cap)

    plt.figure(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=d,
        x="part_recall",
        y="cost_abs_error_capped",
        alpha=0.6,
        s=25,
        color="#E34E4E",
    )
    plt.xlabel("Part Recall")
    plt.ylabel(f"Cost abs error (EUR, capped at p98={cap:.0f})")
    plt.title(f"Cost Error vs Part Recall ({MODEL_NAME})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    out = OUT_GRAPHS / f"{MODEL_NAME}_05_cost_error_vs_parts_recall.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved: {out}")

def print_perfect_prf_json(df):
    perfect = df[(df["part_precision"] == 1.0) & (df["part_recall"] == 1.0)]
    if perfect.empty:
        print("No vehicle with part_precision == 1.0 and part_recall == 1.0 found.")
        return
    for _, row in perfect.iterrows():
        vid = row.get("vehicle_id")
        if not vid:
            continue
        json_path = SCORES_DIR / f"{vid}_score.json"
        if json_path.exists():
            print(f"\n=== {json_path} ===")
            with open(json_path, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print(f"Judge JSON not found for vehicle_id: {vid}")

def run():
    df = load_all_judge_json(SCORES_DIR)
    print(f"Loaded {len(df)} judge files from {SCORES_DIR}")

    save_aggregates(df)
    plot_means(df)
    plot_boxplot_scores(df)
    plot_score_distributions(df)

    # extras
    plot_parts_prf_scatter(df)
    plot_cost_error_vs_parts_recall(df)

    print(f"All graphs saved to: {OUT_GRAPHS}")
    df = load_all_judge_json(SCORES_DIR)
    print(f"Loaded {len(df)} judge files from {SCORES_DIR}")

    save_aggregates(df)
    plot_means(df)
    plot_boxplot_scores(df)
    plot_score_distributions(df)

    # extras
    plot_parts_prf_scatter(df)
    plot_cost_error_vs_parts_recall(df)

    # --- TEMPORARY: print perfect PRF judge JSON ---
    print_perfect_prf_json(df)

    print(f"All graphs saved to: {OUT_GRAPHS}")

if __name__ == "__main__":
    run()