"""
analyze_results.py — Generate tables and graphs for the thesis.

Reads scores from scores/ and outputs/ to produce:
  1. Bar chart: Average judge score per model (EXP-1)
  2. Bar chart: Cost estimation MAE per model
  3. Line chart: Judge score vs. number of images (EXP-2)
  4. Bar chart: API cost per 100 vehicles
  5. Scatter plot: Predicted cost vs. actual cost (best model)

All graphs are saved as .png (for thesis) and .html (for live demo).

Usage:
    python analyze_results.py
    python analyze_results.py --exp 1        # Only EXP-1 graphs
    python analyze_results.py --exp 2        # Only EXP-2 graphs
    python analyze_results.py --exp 3        # Only EXP-3 graphs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Optional plotly for interactive HTML charts
# ---------------------------------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
SCORES = BASE / "scores"
OUTPUTS = BASE / "outputs"
GRAPHS = BASE / "graphs"
DATA = BASE / "data"

MODELS = ["gpt4o", "claude", "gemini"]
MODEL_COLORS = {"gpt4o": "#4CAF50", "claude": "#2196F3", "gemini": "#FF9800"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_fig(fig: plt.Figure, name: str) -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    png_path = GRAPHS / f"{name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  → Saved {png_path}")
    plt.close(fig)


def save_plotly(fig_px, name: str) -> None:
    if not PLOTLY_AVAILABLE:
        return
    GRAPHS.mkdir(parents=True, exist_ok=True)
    html_path = GRAPHS / f"{name}.html"
    fig_px.write_html(str(html_path))
    print(f"  → Saved {html_path}")


def load_comparison() -> pd.DataFrame | None:
    path = SCORES / "comparison.csv"
    if not path.exists():
        print("[warn] scores/comparison.csv not found. Run judge.py first.")
        return None
    return pd.read_csv(path)


def load_outputs_for_model(model: str) -> list[dict]:
    output_dir = OUTPUTS / model
    if not output_dir.exists():
        return []
    results = []
    for f in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            data["_vehicle_id"] = f.stem
            results.append(data)
        except Exception:  # noqa: BLE001
            pass
    return results


def load_ground_truth(vehicle_id: str) -> dict | None:
    gt_path = DATA / "raw" / vehicle_id / "ground_truth.json"
    if gt_path.exists():
        try:
            return json.loads(gt_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    return None


# ---------------------------------------------------------------------------
# EXP-1: Model comparison
# ---------------------------------------------------------------------------


def graph_model_comparison(df: pd.DataFrame) -> None:
    print("\n[EXP-1] Model comparison charts ...")

    # Average judge score per model
    summary = (
        df.groupby("model")[
            ["part_detection_score", "cost_accuracy_score",
             "total_loss_score", "weighted_score"]
        ]
        .mean()
        .reindex(MODELS)
        .dropna()
    )

    # --- Bar chart: weighted score ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary.index,
        summary["weighted_score"],
        color=[MODEL_COLORS.get(m, "#999") for m in summary.index],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Weighted Judge Score (1-5)", fontsize=12)
    ax.set_title("EXP-1: Average Judge Score per Model", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "exp1_model_comparison_score")

    if PLOTLY_AVAILABLE:
        fig_px = px.bar(
            summary.reset_index(),
            x="model",
            y="weighted_score",
            color="model",
            color_discrete_map=MODEL_COLORS,
            title="EXP-1: Average Judge Score per Model",
            labels={"weighted_score": "Weighted Score (1-5)", "model": "Model"},
            text_auto=".2f",
        )
        fig_px.update_layout(yaxis_range=[0, 5.5])
        save_plotly(fig_px, "exp1_model_comparison_score")

    # --- Grouped bar chart: all 3 dimensions ---
    dims = ["part_detection_score", "cost_accuracy_score", "total_loss_score"]
    dim_labels = ["Part Detection", "Cost Accuracy", "Total Loss"]
    x = range(len(summary))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (dim, label) in enumerate(zip(dims, dim_labels)):
        offset = (i - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            summary[dim],
            width=width,
            label=label,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(summary.index))
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("Score (1-5)", fontsize=12)
    ax.set_title("EXP-1: Judge Score by Dimension per Model", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "exp1_model_comparison_dimensions")


# ---------------------------------------------------------------------------
# EXP-1: Cost MAE
# ---------------------------------------------------------------------------


def graph_cost_mae(df: pd.DataFrame) -> None:
    print("\n[EXP-1] Cost MAE chart ...")

    rows = []
    for model in MODELS:
        outputs = load_outputs_for_model(model)
        for o in outputs:
            vid = o.get("_vehicle_id", "")
            gt = load_ground_truth(vid)
            if gt is None:
                continue
            pred_cost = o.get("total_cost_eur", 0) or 0
            actual_cost = gt.get("total_cost_eur", 0) or 0
            if actual_cost > 0:
                rows.append(
                    {"model": model, "mae": abs(pred_cost - actual_cost)}
                )

    if not rows:
        print("  [skip] No ground-truth cost data available yet.")
        return

    mae_df = pd.DataFrame(rows).groupby("model")["mae"].mean().reindex(MODELS).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        mae_df.index,
        mae_df.values,
        color=[MODEL_COLORS.get(m, "#999") for m in mae_df.index],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.bar_label(bars, fmt="%.0f EUR", padding=4, fontsize=11)
    ax.set_ylabel("Mean Absolute Error (EUR)", fontsize=12)
    ax.set_title("EXP-1: Cost Estimation MAE per Model", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "exp1_cost_mae")

    if PLOTLY_AVAILABLE:
        fig_px = px.bar(
            mae_df.reset_index(),
            x="model",
            y="mae",
            color="model",
            color_discrete_map=MODEL_COLORS,
            title="EXP-1: Cost Estimation MAE per Model",
            labels={"mae": "Mean Absolute Error (EUR)", "model": "Model"},
            text_auto=".0f",
        )
        save_plotly(fig_px, "exp1_cost_mae")


# ---------------------------------------------------------------------------
# EXP-2: Image count vs. judge score
# ---------------------------------------------------------------------------


def graph_image_count(df: pd.DataFrame) -> None:
    print("\n[EXP-2] Image count vs. score chart ...")

    # We need scores tagged with image_count — stored in _meta.image_count
    rows = []
    for model in MODELS:
        score_path = SCORES / f"{model}_scores.csv"
        if not score_path.exists():
            continue
        scores_df = pd.read_csv(score_path)
        for _, row in scores_df.iterrows():
            vid = row.get("vehicle_id", "")
            output_file = OUTPUTS / model / f"{vid}.json"
            if not output_file.exists():
                continue
            report = json.loads(output_file.read_text())
            image_count = report.get("_meta", {}).get("image_count", None)
            if image_count is not None:
                rows.append(
                    {
                        "model": model,
                        "image_count": image_count,
                        "weighted_score": row.get("weighted_score", 0),
                    }
                )

    if not rows:
        print("  [skip] No image_count metadata available yet.")
        return

    exp2_df = pd.DataFrame(rows)
    summary = exp2_df.groupby(["model", "image_count"])["weighted_score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODELS:
        subset = summary[summary["model"] == model].sort_values("image_count")
        if subset.empty:
            continue
        ax.plot(
            subset["image_count"],
            subset["weighted_score"],
            marker="o",
            label=model,
            color=MODEL_COLORS.get(model, "#999"),
            linewidth=2,
        )
    ax.set_xlabel("Number of Images per Vehicle", fontsize=12)
    ax.set_ylabel("Average Weighted Score (1-5)", fontsize=12)
    ax.set_title("EXP-2: Judge Score vs. Image Count", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "exp2_image_count_vs_score")

    if PLOTLY_AVAILABLE:
        fig_px = px.line(
            summary,
            x="image_count",
            y="weighted_score",
            color="model",
            color_discrete_map=MODEL_COLORS,
            markers=True,
            title="EXP-2: Judge Score vs. Image Count",
            labels={
                "image_count": "Number of Images per Vehicle",
                "weighted_score": "Average Score (1-5)",
            },
        )
        save_plotly(fig_px, "exp2_image_count_vs_score")


# ---------------------------------------------------------------------------
# API cost chart
# ---------------------------------------------------------------------------


def graph_api_cost() -> None:
    print("\n[COST] API cost per 100 vehicles chart ...")

    # Approximate costs based on problem statement guidance
    cost_data = {
        "gpt4o": 10.0,   # ~$0.01/vehicle * 100
        "claude": 10.0,
        "gemini": 0.0,   # free tier
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(cost_data.keys())
    costs = list(cost_data.values())
    bars = ax.bar(
        models,
        costs,
        color=[MODEL_COLORS.get(m, "#999") for m in models],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.bar_label(bars, fmt="$%.2f", padding=4, fontsize=11)
    ax.set_ylabel("Estimated API Cost (USD per 100 vehicles)", fontsize=12)
    ax.set_title("API Cost per 100 Vehicles", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "api_cost_per_100_vehicles")

    if PLOTLY_AVAILABLE:
        fig_px = px.bar(
            pd.DataFrame({"model": models, "cost_usd": costs}),
            x="model",
            y="cost_usd",
            color="model",
            color_discrete_map=MODEL_COLORS,
            title="API Cost per 100 Vehicles",
            labels={"cost_usd": "Cost (USD)", "model": "Model"},
            text_auto=".2f",
        )
        save_plotly(fig_px, "api_cost_per_100_vehicles")


# ---------------------------------------------------------------------------
# Scatter: predicted vs. actual cost
# ---------------------------------------------------------------------------


def graph_scatter_cost() -> None:
    print("\n[SCATTER] Predicted vs. actual cost ...")

    rows = []
    for model in MODELS:
        outputs = load_outputs_for_model(model)
        for o in outputs:
            vid = o.get("_vehicle_id", "")
            gt = load_ground_truth(vid)
            if gt is None:
                continue
            pred = o.get("total_cost_eur", 0) or 0
            actual = gt.get("total_cost_eur", 0) or 0
            if actual > 0:
                rows.append({"model": model, "predicted": pred, "actual": actual})

    if not rows:
        print("  [skip] No ground-truth cost data available yet.")
        return

    scatter_df = pd.DataFrame(rows)
    max_val = max(scatter_df[["predicted", "actual"]].max()) * 1.1

    fig, ax = plt.subplots(figsize=(8, 8))
    for model in MODELS:
        sub = scatter_df[scatter_df["model"] == model]
        if sub.empty:
            continue
        ax.scatter(
            sub["actual"],
            sub["predicted"],
            label=model,
            color=MODEL_COLORS.get(model, "#999"),
            alpha=0.7,
            s=60,
        )
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="Perfect prediction")
    ax.set_xlabel("Actual Repair Cost (EUR)", fontsize=12)
    ax.set_ylabel("Predicted Repair Cost (EUR)", fontsize=12)
    ax.set_title("Predicted vs. Actual Repair Cost", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "scatter_predicted_vs_actual_cost")

    if PLOTLY_AVAILABLE:
        fig_px = px.scatter(
            scatter_df,
            x="actual",
            y="predicted",
            color="model",
            color_discrete_map=MODEL_COLORS,
            title="Predicted vs. Actual Repair Cost",
            labels={
                "actual": "Actual Repair Cost (EUR)",
                "predicted": "Predicted Repair Cost (EUR)",
            },
        )
        fig_px.add_trace(
            go.Scatter(
                x=[0, scatter_df["actual"].max()],
                y=[0, scatter_df["actual"].max()],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash", "color": "black"},
            )
        )
        save_plotly(fig_px, "scatter_predicted_vs_actual_cost")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Vehicle Damage Assessment — Analyze Results & Generate Graphs"
    )
    parser.add_argument(
        "--exp",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which experiment graphs to generate (default: all)",
    )
    args = parser.parse_args()

    GRAPHS.mkdir(parents=True, exist_ok=True)

    df = load_comparison()

    if args.exp in ("1", "all"):
        if df is not None:
            graph_model_comparison(df)
            graph_cost_mae(df)
        graph_api_cost()

    if args.exp in ("2", "all"):
        if df is not None:
            graph_image_count(df)

    if args.exp in ("3", "all"):
        # EXP-3 uses the same score comparison but with prompt-style tag
        # (when judge.py is re-run after --prompt simple/detailed experiments)
        print("\n[EXP-3] Prompt comparison — re-run judge.py after EXP-3 experiments")

    # Scatter is always included (best-model quality check)
    if args.exp == "all":
        graph_scatter_cost()

    print("\nDone. Check the graphs/ folder.")


if __name__ == "__main__":
    main()
