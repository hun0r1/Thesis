"""
judge.py — LLM judge scoring for AI Vehicle Damage Assessment.

Scores each model's output against ground truth on three dimensions:
  1. Part detection (35%)
  2. Cost accuracy  (35%)
  3. Total loss decision (30%)

Each dimension is scored 1-5 by Gemini (free tier — no extra cost).

Usage:
    python judge.py                        # Score all models
    python judge.py --model claude         # Score one model
    python judge.py --vehicle demo_car     # Score a single vehicle
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
OUTPUTS = BASE / "outputs"
DATA = BASE / "data"
SCORES = BASE / "scores"

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are an expert automotive insurance auditor evaluating AI damage reports.

GROUND TRUTH:
{ground_truth}

AI REPORT:
{ai_report}

Score the AI report on three dimensions. Return ONLY valid JSON, no explanation:
{{
  "part_detection_score": <1-5>,
  "part_detection_reason": "<one sentence>",
  "cost_accuracy_score": <1-5>,
  "cost_accuracy_reason": "<one sentence>",
  "total_loss_score": <1-5>,
  "total_loss_reason": "<one sentence>",
  "weighted_score": <float 1-5>
}}

Scoring rubric:
- part_detection: 1=completely wrong parts, 3=some correct, 5=all parts correctly identified
- cost_accuracy: 1=>50% off, 3=within 25%, 5=within 10% of ground truth cost
- total_loss: 1=wrong verdict with no reason, 3=correct verdict, 5=correct verdict with clear justification
- weighted_score: (part_detection * 0.35) + (cost_accuracy * 0.35) + (total_loss * 0.30)

If no ground truth is available, score based on internal consistency and plausibility.
"""

# ---------------------------------------------------------------------------
# Gemini judge caller
# ---------------------------------------------------------------------------


def call_gemini_judge(prompt: str) -> dict:
    import google.generativeai as genai  # noqa: PLC0415

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def load_ground_truth(vehicle_id: str) -> dict | None:
    gt_path = DATA / "raw" / vehicle_id / "ground_truth.json"
    if gt_path.exists():
        return json.loads(gt_path.read_text())
    return None


def score_vehicle(vehicle_id: str, model: str, ai_report: dict) -> dict:
    ground_truth = load_ground_truth(vehicle_id)
    gt_str = json.dumps(ground_truth, indent=2) if ground_truth else "NOT AVAILABLE"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=gt_str,
        ai_report=json.dumps(ai_report, indent=2),
    )

    try:
        scores = call_gemini_judge(prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] judge failed for {vehicle_id}/{model}: {exc}")
        return {}

    scores["vehicle_id"] = vehicle_id
    scores["model"] = model
    scores["has_ground_truth"] = ground_truth is not None
    return scores


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------


def score_model(model: str, vehicle_filter: str | None) -> list[dict]:
    output_dir = OUTPUTS / model
    if not output_dir.exists():
        print(f"[warn] No outputs found for model: {model}")
        return []

    if vehicle_filter:
        output_files = [output_dir / f"{vehicle_filter}.json"]
        output_files = [f for f in output_files if f.exists()]
    else:
        output_files = sorted(output_dir.glob("*.json"))

    if not output_files:
        print(f"[warn] No output files found for {model}")
        return []

    rows = []
    print(f"\n=== Judging: {model} ({len(output_files)} vehicles) ===")
    for output_file in output_files:
        vehicle_id = output_file.stem
        ai_report = json.loads(output_file.read_text())
        print(f"  [judge] {vehicle_id} ...")
        row = score_vehicle(vehicle_id, model, ai_report)
        if row:
            rows.append(row)
        time.sleep(0.5)  # be polite to the free-tier API

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Vehicle Damage Assessment — LLM Judge Scorer"
    )
    parser.add_argument(
        "--model",
        choices=["claude", "gpt4o", "gemini"],
        default=None,
        help="Which model's outputs to score (default: all)",
    )
    parser.add_argument(
        "--vehicle",
        default=None,
        help="Score a single vehicle only",
    )
    args = parser.parse_args()

    SCORES.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else ["claude", "gpt4o", "gemini"]
    all_rows = []

    for model in models:
        rows = score_model(model, args.vehicle)
        all_rows.extend(rows)

        if rows:
            df_model = pd.DataFrame(rows)
            out_path = SCORES / f"{model}_scores.csv"
            df_model.to_csv(out_path, index=False)
            print(f"  → Saved {len(rows)} scores to {out_path}")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        comparison_path = SCORES / "comparison.csv"
        df_all.to_csv(comparison_path, index=False)
        print(f"\n→ Combined comparison saved to {comparison_path}")

        # Print summary table
        summary = (
            df_all.groupby("model")[
                ["part_detection_score", "cost_accuracy_score",
                 "total_loss_score", "weighted_score"]
            ]
            .mean()
            .round(2)
        )
        print("\n=== Summary (mean scores) ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
