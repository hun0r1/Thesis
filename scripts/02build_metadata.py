"""
STEP 2 — Build metadata.csv from your existing vehicle folders
--------------------------------------------------------------
Point RAW_DATA_DIR at your existing scraped dataset folder.
The script scans every subfolder, reads ground_truth.json if present,
and writes data/metadata/metadata.csv — one row per vehicle.

Usage:
    python 02_build_metadata.py
"""

import csv
import json
from pathlib import Path

# ── CONFIGURE THESE ────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(r"C:\Users\Hunor\vehicle-damage-ai")
RAW_DATA_DIR  = Path(r"C:\Users\Hunor\existing-dataset")  #  scraped data
# ───────────────────────────────────────────────────────────────────────────

OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"

FIELDNAMES = [
    "vehicle_id",           # original folder name — never renamed
    "folder_path",          # absolute path to the vehicle folder
    "image_count",          # how many jpg/png files found
    "has_ground_truth",     # ground_truth.json exists?
    "has_damage_gt",        # meta.has_damage_ground_truth
    "has_cost_gt",          # meta.has_cost_ground_truth
    "has_total_loss_gt",    # meta.has_total_loss_ground_truth
    "source_origin",        # source.origin from GT json
    "make",                 # vehicle.make
    "model",                # vehicle.model
    "year",                 # vehicle.year
    "damage_category",      # labels.damage_category
    "overall_severity",     # labels.overall_severity
    "insurer_total_loss",   # economics.insurer_total_loss
    "split",                # train / test — fill 'test' for GT vehicles, 'unused' for rest
    "notes",                # any manual notes you want to add
]

def scan_vehicle(folder: Path) -> dict:
    row = {f: "" for f in FIELDNAMES}
    row["vehicle_id"]   = folder.name
    row["folder_path"]  = str(folder)
    row["split"]        = "unused"
    row["notes"]        = ""

    # Count images
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
    row["image_count"] = len(images)

    # Read ground truth if it exists
    gt_file = folder / "groundTruthSchema.json"
    if gt_file.exists():
        row["has_ground_truth"] = "true"
        row["split"] = "test"   # default GT vehicles to test set
        try:
            gt = json.loads(gt_file.read_text(encoding="utf-8"))
            row["source_origin"]       = gt.get("source", {}).get("origin", "")
            row["make"]                = gt.get("vehicle", {}).get("make", "")
            row["model"]               = gt.get("vehicle", {}).get("model", "")
            row["year"]                = gt.get("vehicle", {}).get("year", "")
            row["damage_category"]     = gt.get("labels", {}).get("damage_category", "")
            row["overall_severity"]    = gt.get("labels", {}).get("overall_severity", "")
            row["insurer_total_loss"]  = gt.get("economics", {}).get("insurer_total_loss", "")
            meta = gt.get("meta", {})
            row["has_damage_gt"]       = meta.get("has_damage_ground_truth", "")
            row["has_cost_gt"]         = meta.get("has_cost_ground_truth", "")
            row["has_total_loss_gt"]   = meta.get("has_total_loss_ground_truth", "")
        except Exception as e:
            row["notes"] = f"JSON parse error: {e}"
    else:
        row["has_ground_truth"] = "false"

    return row

def build_metadata():
    if not RAW_DATA_DIR.exists():
        print(f"ERROR: RAW_DATA_DIR not found: {RAW_DATA_DIR}")
        return

    vehicle_folders = [f for f in RAW_DATA_DIR.iterdir() if f.is_dir()]
    print(f"Found {len(vehicle_folders)} vehicle folders in {RAW_DATA_DIR}")

    rows = []
    gt_count = 0
    for folder in sorted(vehicle_folders):
        row = scan_vehicle(folder)
        rows.append(row)
        if row["has_ground_truth"] == "true":
            gt_count += 1

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"  Vehicles with ground truth (test set): {gt_count}")
    print(f"  Vehicles without ground truth (unused): {len(rows) - gt_count}")
    print("\nNext: open metadata.csv, review, then run 03_analyze.py")

if __name__ == "__main__":
    build_metadata()