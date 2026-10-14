"""
STEP 4b — Smart Judge Scoring (Pure Python, No LLM)
----------------------------------------------------
Compares groundTruthSchema.json vs aiSchema.json for each vehicle.
Everything is calculated by Python — no API calls, no cost.

Normalizes:
  - part names   using parts_mapping.csv  (part_norm -> part_group)
  - severity     using built-in SEVERITY_MAP

Scoring (1-5 scale):
  score_parts     → part recall (how many GT parts did AI find?)
  score_severity  → severity match rate on matched parts
  score_cost      → tolerance-based cost accuracy from economics.estimated_repair_cost
  score_overall   → average of available component scores (parts + severity + cost)

Usage:
    python 04smart_judge.py --limit 5    # test on 5 first
    python 04smart_judge.py              # full run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

# ── CONFIGURE ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\Hunor\EGYETEM\A3.felev\Allamvizsga+tesztek\AutokReportok")
MAPPING_CSV  = PROJECT_ROOT / "parts_mapping.csv"

GT_FILENAMES = ["groundTruthSchema.json", "groundtruthschema.json"]
AI_FILENAMES = ["aiSchema.json", "aischema.json"]

MODEL_NAME   = "claude"   # used only for output folder name
# ─────────────────────────────────────────────────────────────────────────────

METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"

# ── COST SCORING CONFIG ──────────────────────────────────────────────────────
COST_ABS_TOLERANCE = 100.0  # EUR: abs error <= 100 counts as perfect
# After tolerance, bucket by relative error vs GT cost
# (Tune these if you want stricter/looser scoring)
COST_REL_BUCKETS = [
    (0.10, 4),  # <=10% error
    (0.20, 3),  # <=20% error
    (0.35, 2),  # <=35% error
    (1.00, 1),  # >35% error
]
# ─────────────────────────────────────────────────────────────────────────────

# ── SEVERITY NORMALIZATION ───────────────────────────────────────────────────
SEVERITY_MAP = {
    "heavy":    "heavy",
    "severe":   "heavy",
    "moderate": "moderate",
    "medium":   "moderate",
    "minor":    "minor",
    "light":    "minor",
    "inspect":  "minor",
    "deployed": "minor",
    "none":     "none",
    "unknown":  "none",
    "n/a":      "none",
    "na":       "none",
}

def norm_severity(value: Any) -> str:
    if value is None:
        return "none"
    key = str(value).lower().strip()
    return SEVERITY_MAP.get(key, key)


# ── PART NAME NORMALIZATION ──────────────────────────────────────────────────
def build_part_normalizer(mapping_csv: Path) -> dict[str, str]:
    """
    Expects columns:
      - part_norm
      - part_group
    """
    if not mapping_csv.exists():
        print(f"WARNING: {mapping_csv} not found — parts will not be normalized.")
        return {}

    df = pd.read_csv(mapping_csv)
    if "part_norm" not in df.columns or "part_group" not in df.columns:
        raise ValueError(f"{mapping_csv} must contain columns: part_norm, part_group")

    return dict(zip(
        df["part_norm"].astype(str).str.lower().str.strip(),
        df["part_group"].astype(str).str.lower().str.strip()
    ))

def norm_part(name: Any, normalizer: dict[str, str]) -> str:
    key = str(name or "").lower().strip().replace(" ", "_").replace("-", "_")
    return normalizer.get(key, key)


# ── SCORING THRESHOLDS ───────────────────────────────────────────────────────
def rate_to_score(rate: float) -> int:
    """Convert a 0-1 match rate to a 1-5 score."""
    if rate >= 0.90: return 5
    if rate >= 0.70: return 4
    if rate >= 0.50: return 3
    if rate >= 0.30: return 2
    return 1

def cost_error_to_score(gt_cost: float | None, ai_cost: float | None) -> tuple[int | None, float | None]:
    """
    Returns: (score_cost, abs_error)

    Absolute-error thresholds (EUR):
      <= 100   -> 5
      <= 200   -> 4
      <= 800   -> 3
      <= 1000  -> 2
      > 1000   -> 1
    """
    if not isinstance(gt_cost, (int, float)) or not isinstance(ai_cost, (int, float)):
        return None, None

    gt = float(gt_cost)
    ai = float(ai_cost)

    if gt <= 0:
        return None, None

    abs_error = abs(ai - gt)

    if abs_error <= 100:
        return 5, abs_error
    if abs_error <= 200:
        return 4, abs_error
    if abs_error <= 800:
        return 3, abs_error
    if abs_error <= 1000:
        return 2, abs_error
    return 1, abs_error


# ── CORE COMPARISON ──────────────────────────────────────────────────────────
def extract_parts(data: dict, normalizer: dict[str, str]) -> list[dict[str, str]]:
    """Extract and normalize parts list from a JSON schema."""
    raw = (data.get("damage") or {}).get("parts") or []
    result: list[dict[str, str]] = []
    for p in raw:
        result.append({
            "part":     norm_part(p.get("part", ""), normalizer),
            "severity": norm_severity(p.get("severity")),
        })
    return result


def compare(gt_data: dict, ai_data: dict, normalizer: dict[str, str]) -> dict[str, Any]:
    """Full Python comparison between GT and AI output. No LLM needed."""

    gt_parts = extract_parts(gt_data, normalizer)
    ai_parts = extract_parts(ai_data, normalizer)

    gt_names = {p["part"] for p in gt_parts if p.get("part")}
    ai_names = {p["part"] for p in ai_parts if p.get("part")}

    # ── Part detection ───────────────────────────────────────────────────
    matched   = gt_names & ai_names
    missed    = sorted(gt_names - ai_names)
    extra     = sorted(ai_names - gt_names)
    gt_count  = len(gt_names)
    ai_count  = len(ai_names)

    precision = len(matched) / ai_count if ai_count > 0 else 0.0
    recall    = len(matched) / gt_count if gt_count > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    score_parts = rate_to_score(recall) if gt_count > 0 else None

    # ── Severity match on matched parts ─────────────────────────────────
    gt_sev = {p["part"]: p["severity"] for p in gt_parts}
    ai_sev = {p["part"]: p["severity"] for p in ai_parts}

    sev_matches = 0
    sev_total   = 0
    for part in matched:
        gt_s = gt_sev.get(part, "none")
        ai_s = ai_sev.get(part, "none")

        # only compare if GT has a meaningful label
        if gt_s != "none":
            sev_total += 1
            if gt_s == ai_s:
                sev_matches += 1

    sev_rate       = (sev_matches / sev_total) if sev_total > 0 else None
    score_severity = rate_to_score(sev_rate) if sev_rate is not None else None

    # ── Cost score ──────────────────────────────────────────────────────
    gt_cost = (gt_data.get("economics") or {}).get("estimated_repair_cost")
    ai_cost = (ai_data.get("economics") or {}).get("estimated_repair_cost")

    score_cost, cost_abs_error = cost_error_to_score(gt_cost, ai_cost)
    cost_rel_error = None  # optional: remove completely if you don't want it
    # ── Overall severity label (optional check) ─────────────────────────
    gt_overall_sev = norm_severity((gt_data.get("labels") or {}).get("overall_severity", ""))
    ai_overall_sev = norm_severity((ai_data.get("labels") or {}).get("overall_severity", ""))
    overall_sev_match = (gt_overall_sev == ai_overall_sev) if gt_overall_sev != "none" else None

    # ── Total loss (optional check) ─────────────────────────────────────
    # NOTE: your GT uses insurer_total_loss sometimes; AI uses predicted_total_loss.
    gt_tl = (gt_data.get("economics") or {}).get("insurer_total_loss")
    ai_tl = (ai_data.get("economics") or {}).get("predicted_total_loss")
    total_loss_match = (bool(gt_tl) == bool(ai_tl)) if (gt_tl is not None and ai_tl is not None) else None

    # ── Overall score: average of available component scores ─────────────
    components = [s for s in [score_parts, score_severity, score_cost] if s is not None]
    score_overall = round(sum(components) / len(components)) if components else None

    return {
        # Scores 1-5
        "score_parts":              score_parts,
        "score_severity":           score_severity,
        "score_cost":               score_cost,
        "score_overall":            score_overall,

        # Part metrics 0-1
        "part_precision":           round(precision, 3),
        "part_recall":              round(recall, 3),
        "part_f1":                  round(f1, 3),
        "gt_part_count":            gt_count,
        "ai_part_count":            ai_count,
        "matched_part_count":       len(matched),

        # Severity metrics
        "severity_match_rate":      round(sev_rate, 3) if sev_rate is not None else None,
        "severity_matches":         sev_matches,
        "severity_total_compared":  sev_total,

        # Cost metrics
        "gt_estimated_repair_cost": float(gt_cost) if isinstance(gt_cost, (int, float)) else None,
        "ai_estimated_repair_cost": float(ai_cost) if isinstance(ai_cost, (int, float)) else None,
        "cost_abs_error":           round(cost_abs_error, 2) if cost_abs_error is not None else None,
        "cost_rel_error":           round(cost_rel_error, 4) if cost_rel_error is not None else None,

        # Label checks
        "overall_severity_match":   overall_sev_match,
        "total_loss_match":         total_loss_match,

        # Detail lists
        "matched_parts":            sorted(matched),
        "missed_parts":             missed,
        "extra_parts":              extra,
    }


# ── PER-VEHICLE RUNNER ───────────────────────────────────────────────────────
def first_existing(folder: Path, names: list[str]) -> Path | None:
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None


def judge_vehicle(vehicle_id: str, folder: Path, normalizer: dict[str, str]) -> dict | None:
    gt_path = first_existing(folder, GT_FILENAMES)
    ai_path = first_existing(folder, AI_FILENAMES)

    if not gt_path:
        print(f"  [SKIP] {vehicle_id} — no ground truth JSON ({', '.join(GT_FILENAMES)})")
        return None
    if not ai_path:
        print(f"  [SKIP] {vehicle_id} — no AI JSON ({', '.join(AI_FILENAMES)})")
        return None

    try:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ai = json.loads(ai_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [ERROR] {vehicle_id} — JSON read failed: {e}")
        return None

    result = compare(gt, ai, normalizer)
    result["vehicle_id"] = vehicle_id
    result["gt_file"] = gt_path.name
    result["ai_file"] = ai_path.name
    return result


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run(limit: int | None = None):
    scores_dir = PROJECT_ROOT / "scores" / MODEL_NAME
    scores_dir.mkdir(parents=True, exist_ok=True)

    normalizer = build_part_normalizer(MAPPING_CSV)
    print(f"Part mappings loaded : {len(normalizer)}")

    df = pd.read_csv(METADATA_CSV)
    test_df = df[df["split"] == "test"].copy()
    if limit:
        test_df = test_df.head(limit)

    print(f"Vehicles to judge    : {len(test_df)}")
    print("-" * 55)

    all_scores = []

    for _, row in test_df.iterrows():
        vid        = row["vehicle_id"]
        folder     = Path(row["folder_path"])
        score_file = scores_dir / f"{vid}_score.json"

        if score_file.exists():
            print(f"  [SKIP]  {vid}")
            all_scores.append(json.loads(score_file.read_text(encoding="utf-8")))
            continue

        print(f"  [JUDGE] {vid}")
        result = judge_vehicle(vid, folder, normalizer)

        if result:
            score_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            all_scores.append(result)

    if not all_scores:
        print("\nNo scores collected.")
        return

    # ── Summary CSV ──────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(all_scores)
    summary_path = PROJECT_ROOT / "scores" / f"{MODEL_NAME}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results | {MODEL_NAME.upper()} | {len(summary_df)} vehicles scored")
    print(f"{'='*55}")

    print("\n  Scores (1-5 scale):")
    for col, label in [
        ("score_parts",    "Part Detection (recall)"),
        ("score_severity", "Severity Accuracy"),
        ("score_cost",     f"Cost Accuracy (tol={COST_ABS_TOLERANCE:.0f})"),
        ("score_overall",  "Overall"),
    ]:
        if col in summary_df:
            avg = pd.to_numeric(summary_df[col], errors="coerce").mean()
            print(f"    {label:30s}: {avg:.2f} / 5.0")

    print("\n  Raw metrics:")
    for col, label in [
        ("part_f1",             "Part F1"),
        ("part_precision",      "Part Precision"),
        ("part_recall",         "Part Recall"),
        ("severity_match_rate", "Severity Match Rate"),
        ("cost_abs_error",      "Cost Abs Error (EUR)"),
    ]:
        if col in summary_df:
            avg = pd.to_numeric(summary_df[col], errors="coerce").mean()
            print(f"    {label:30s}: {avg:.3f}")

    print(f"\nSaved : {summary_path}")
    print(f"Next  : python 05_results.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N vehicles for testing")
    args = parser.parse_args()
    run(args.limit)