
import csv
from pathlib import Path

RAW_DATA_DIR = Path(r"D:\Hunor\EGYETEM\A3.felev\Allamvizsga+tesztek\AutokReportok\SCA_Auction")
OUTPUT_CSV = Path(r"D:\Hunor\EGYETEM\A3.felev\Allamvizsga+tesztek\vehicle-damage-ai\data\metadata\metadata.csv")

def update_image_counts():
    # Build a mapping: vehicle_id -> new image count (including .webp)
    image_counts = {}
    for folder in RAW_DATA_DIR.iterdir():
        if folder.is_dir():
            count = sum(1 for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp') for _ in folder.glob(ext))
            image_counts[folder.name] = count

    # Read and update metadata.csv
    rows = []
    with open(OUTPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            vid = row.get("vehicle_id", "")
            if vid in image_counts:
                row["image_count"] = image_counts[vid]
            rows.append(row)

    # Write back the updated CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated image_count for {len(image_counts)} vehicles in {OUTPUT_CSV}")

if __name__ == "__main__":
    update_image_counts()