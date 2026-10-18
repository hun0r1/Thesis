"""
STEP 4 — Judge Scoring (Gemini 2.5 Flash)
-----------------------------------------
Reads groundTruthSchema.json and aiSchema.json from each vehicle folder,
uses Gemini 2.5 Flash to score the comparison, and writes judge.json
back into the SAME vehicle folder.

Folder layout:
your-dataset/<source>/<vehicle_id>/
  - groundTruthSchema.json
  - aiSchema.json
  - judge.json   (created by this script)

Also writes a global summary CSV to:
  <PROJECT_ROOT>/scores/judge_summary.csv

Usage:
    pip install google-generativeai pandas python-dotenv
    python 04_judge.py --limit 5
    python 04_judge.py
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

# ── CONFIGURE THIS ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\Hunor\EGYETEM\A3.felev\Allamvizsga+tesztek\AutokReportok")
# ──────────────────────────────────────────────────────────────────────────

JUDGE_MODEL_ID = "gemini-2.5-flash"   # judge model is fixed

METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
GT_FILENAME = "groundTruthSchema.json"
AI_FILENAME = "aiSchema.json"
JUDGE_FILENAME = "judge.json"

load_dotenv(PROJECT_ROOT / ".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

JUDGE_PROMPT = """You are an impartial evaluator for vehicle damage assessment outputs.

Task:
Compare the AI-generated report to the ground truth report. Both are JSON objects.

Rules:
- Focus on factual correctness and completeness.
- Ignore wording/formatting differences.
- Do not reward extra claims not supported by the ground truth.
- Only score a dimension if the ground truth contains enough data for it; otherwise output null for that score.
- Use integers 1,2,3,4,5 for scores (or null).

Scoring rubric:
5 = Fully correct / equivalent to ground truth
4 = Mostly correct, minor omissions
3 = Partially correct, missing key info
2 = Mostly incorrect
1 = Completely incorrect

Ground truth (reference):
{reference_answer}

AI output:
{llm_answer}

Scoring requirements:
- parts: compare damage.parts[*].part (+ location if present)
- severity: compare damage.parts[*].severity and labels.overall_severity
- repair_action: compare damage.parts[*].repair_action
- cost: compare economics.estimated_repair_cost (numeric). If ground truth cost is null/missing, output null.
- total_loss: compare economics.predicted_total_loss (AI) vs economics.predicted_total_loss (GT). If GT is null/missing, output null.
- If the ground truth has costs missing or null, you MUST ignore and NOT mention that aischema.json did costs in the explanation anyways.
- Be slightly generous: When unsure between two scores, choose the higher score.

Output ONLY valid JSON with exactly these keys:
{{
  "parts": <1-5 or null>,
  "severity": <1-5 or null>,
  "repair_action": <1-5 or null>,
  "cost": <1-5 or null>,
  "total_loss": <1-5 or null>,
  "overall": <1-5 or null>,
  "explanation": "<one sentence>"
}}
"""


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()
        else:
            t = t.lstrip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def _to_bool_or_none(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def _to_number_or_none(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _has_parts_data(gt_relevant: dict) -> bool:
    parts = (gt_relevant.get("damage") or {}).get("parts")
    return isinstance(parts, list) and len(parts) > 0


def _has_cost_data(gt_relevant: dict) -> bool:
    cost = (gt_relevant.get("economics") or {}).get("estimated_repair_cost")
    return _to_number_or_none(cost) is not None


def _has_total_loss_data(gt_relevant: dict) -> bool:
    tl = (gt_relevant.get("economics") or {}).get("predicted_total_loss")
    return _to_bool_or_none(tl) is not None


def judge_vehicle(vehicle_id: str, folder_path: Path) -> dict | None:
    gt_path = folder_path / GT_FILENAME
    ai_path = folder_path / AI_FILENAME
    out_path = folder_path / JUDGE_FILENAME

    if not gt_path.exists():
        print(f"  [SKIP] {vehicle_id} — missing {GT_FILENAME}")
        return None
    if not ai_path.exists():
        print(f"  [SKIP] {vehicle_id} — missing {AI_FILENAME}")
        return None

    # Skip if already judged
    if out_path.exists():
        print(f"  [SKIP] {vehicle_id} — already has {JUDGE_FILENAME}")
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    try:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ai = json.loads(ai_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [ERROR] {vehicle_id} — JSON read failed: {e}")
        return None

    gt_relevant = {
        "damage": gt.get("damage", None),
        "economics": gt.get("economics", None),
        "labels": gt.get("labels", None),
    }
    ai_relevant = {
        "damage": ai.get("damage", None),
        "economics": ai.get("economics", None),
        "labels": ai.get("labels", None),
    }

    prompt = JUDGE_PROMPT.format(
        reference_answer=json.dumps(gt_relevant, indent=2, ensure_ascii=False),
        llm_answer=json.dumps(ai_relevant, indent=2, ensure_ascii=False),
    )

    try:
        model = genai.GenerativeModel(JUDGE_MODEL_ID)
        response = model.generate_content(prompt)
        raw = _strip_code_fences(response.text)
        scores = json.loads(raw)

        # Enforce null where GT lacks data for a dimension
        if not _has_parts_data(gt_relevant):
            scores["parts"] = None
            scores["severity"] = None
            scores["repair_action"] = None
        if not _has_cost_data(gt_relevant):
            scores["cost"] = None
        if not _has_total_loss_data(gt_relevant):
            scores["total_loss"] = None

        judge_obj = {
            "vehicle_id": vehicle_id,
            "judge_model": JUDGE_MODEL_ID,
            "scores": {
                "parts": scores.get("parts"),
                "severity": scores.get("severity"),
                "repair_action": scores.get("repair_action"),
                "cost": scores.get("cost"),
                "total_loss": scores.get("total_loss"),
                "overall": scores.get("overall"),
            },
            "explanation": scores.get("explanation", ""),
        }

        out_path.write_text(json.dumps(judge_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return judge_obj

    except Exception as e:
        print(f"  [ERROR] {vehicle_id} — judge failed: {e}")
        return None


def run(limit: int | None = None):
    df = pd.read_csv(METADATA_CSV)
    test_df = df[df["split"] == "test"].copy()

    if limit:
        test_df = test_df.head(limit)

    print(f"\nJudge scoring | judge: {JUDGE_MODEL_ID} | vehicles: {len(test_df)}")
    print("-" * 80)

    all_scores = []
    for _, row in test_df.iterrows():
        vid = str(row["vehicle_id"])
        folder = Path(row["folder_path"])

        if not folder.exists():
            print(f"  [MISS] {vid} — folder not found: {folder}")
            continue

        print(f"  [JUDGE] {vid}")
        res = judge_vehicle(vid, folder)
        if res:
            # Flatten for summary
            flat = {"vehicle_id": vid, "judge_model": JUDGE_MODEL_ID, "folder_path": str(folder)}
            scores = (res.get("scores") or {})
            for k, v in scores.items():
                flat[k] = v
            flat["explanation"] = res.get("explanation", "")
            all_scores.append(flat)

        time.sleep(1)  # Gemini free tier / rate limiting

    # Optional global summary CSV (recommended for graphs)
    if all_scores:
        scores_dir = PROJECT_ROOT / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        summary_path = scores_dir / "judge_summary.csv"
        pd.DataFrame(all_scores).to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")

    print("\nDone. Each vehicle folder now contains judge.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)