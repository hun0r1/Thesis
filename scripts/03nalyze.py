import os
import time
from google import genai
from google.genai import types
import mimetypes
mimetypes.add_type('image/webp', '.webp')
# Initialize the Gemini client (It automatically picks up the GOOGLE_API_KEY environment variable)
client = genai.Client()

# This is the exact prompt and persona used in our chat
SYSTEM_INSTRUCTION = """
Purpose and Goals:
* Act as an 'Image Car Damage Assessment AI' specialized in generating ground truth schemas from user-provided images of vehicle damage.
* Analyze images to identify specific damage components, severity, and location.
* Output data in a structured JSON format that mirrors the provided ground truth example.

Behaviors and Rules:
1) Image Processing and Analysis:
a) Upon receiving an image, meticulously scan for exterior and interior vehicle damage.
b) Evaluate the severity of the damage for each identified part (e.g., 'minor', 'moderate', 'severe').
c) Determine the specific car part affected based on standard automotive terminology.
d) If you also get a txt file with extra information you may integrate that too.

2) Schema Generation:
a) Produce a JSON output for every assessment.
b) Ensure the structure strictly follows the ground truth schema.
c) If a field in the schema is not applicable or visible in the image, use empty strings ('') or boolean 'false' as appropriate.

Overall Tone:
* Professional, technical, and objective.
* Efficient and data-driven.
* Precise in terminology related to automotive engineering and insurance claims.

Always set the 'annotated_by' value in meta to 'hun0r1' in the output schema.

"""

def process_vehicle_folder(folder_path, vehicle_index):
    print(f"\n--- Processing Vehicle Folder: {folder_path} ---")

    # ── SKIP LOGIC ────────────────────────────────────────────────────────
    final_gt = os.path.join(folder_path, "ground_truth.json")
    draft_gt = os.path.join(folder_path, "aischema.json")

    if os.path.exists(final_gt):
        print(f"  [SKIP] Already has ground_truth.json — skipping.")
        return
    if os.path.exists(draft_gt):
        print(f"  [SKIP] Already has aischema.json — skipping.")
        return
    # ─────────────────────────────────────────────────────────────────────

    files_to_upload = []

    # Collect all image and txt files in the current folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        ext = filename.lower()
        if ext.endswith(('.jpg', '.jpeg', '.png', '.txt','.webp')):
            files_to_upload.append(filepath)

    if not files_to_upload:
        print("No valid images or text files found. Skipping.")
        return

    uploaded_files = []
    try:
        # Upload the files to the Gemini API
        for filepath in files_to_upload:
            print(f"Uploading {os.path.basename(filepath)}...")
            uploaded_file = client.files.upload(file=filepath)
            uploaded_files.append(uploaded_file)




        folder_name = os.path.basename(folder_path)

        schema_example = """
{
  "vehicle_id": "FOLDER_NAME",
  "source": {"origin": "direct|ecartrade|totalcar|sca_auction", "publisher": null, "url": null, "language": "unknown", "reported_at": null},
  "vehicle": {
    "year": null, "make": "detected or null", "model": "detected or null",
    "generation": null, "body_style": "sedan|hatchback|suv|pickup|van|coupe|other|unknown",
    "engine": null, "drivetrain": null, "vin": null,
    "usage_history": "private|fleet|taxi|rental|unknown", "odometer_km": null, "market_region": "EU"
  },
  "accident": {
    "date": null,
    "location": {"country": null, "city": null},
    "collision_type": "head_on|rear_end|side|front_impact|rollover|multi_vehicle|other|unknown",
    "impact_zone": "front|front_left|front_right|rear|rear_left|rear_right|left_side|right_side|multiple|unknown",
    "description": "one sentence describing damage pattern",
    "other_vehicle": null
  },
  "damage": {
    "airbags_deployed": {"front_driver": null, "front_passenger": null, "side": null, "curtain": null},
    "structural": {
      "frame": "none|minor|moderate|severe|unknown",
      "a_pillar": "none|minor|moderate|severe|unknown",
      "b_pillar": "none|minor|moderate|severe|unknown",
      "floor_pan": "none|minor|moderate|severe|unknown"
    },
    "parts": [
      {
        "part": "part_name_in_snake_case",
        "location": "front|front_left|front_right|rear|rear_left|rear_right|left_side|right_side|roof|interior|underbody|unknown",
        "category": "body|suspension|lighting|safety|powertrain|glass|interior|electronics|other",
        "severity": "none|minor|moderate|heavy|unknown",
        "repair_action": "replace|repair|repaint|inspect|none|unknown",
        "estimated_cost_eur": null,
        "labor_hours": null
      }
    ]
  },
  "economics": {
    "currency": "EUR",
    "estimated_repair_cost": null,
    "market_value_pre_accident": null,
    "insurer_total_loss": null,
    "predicted_total_loss": null
  },
  "labels": {
    "damage_category": "front_collision|rear_collision|side_collision|rollover|multi_zone|minor_damage|unknown",
    "overall_severity": "mild|moderate|severe|total_loss|unknown",
    "drivable_after_accident": null
  },
  "meta": {
    "annotated_by": "ai",
    "annotated_at": "2026-01-01T00:00:00Z",
    "confidence": "ai-generated"
    "images_used": 0,
    "has_damage_ground_truth": false,
    "has_cost_ground_truth": false,
    "has_total_loss_ground_truth": false
  }
}"""

        prompt = (
            f"Analyze these vehicle images and generate a ground truth JSON schema.\n"
            f"Vehicle folder name: {folder_name}\n"
            f"Set vehicle_id to exactly: {folder_name}\n\n"
            f"Follow this exact schema structure:\n{schema_example}\n\n"
        )

        prompt += (
                "For each part in damage.parts, fill in estimated_cost_eur with a realistic price"
                "European bodyshop repair cost in EUR for that specific part and repair action. "
                "Use realistic reference ranges, do not put total loss unless you are entirely sure from the images "
                "Also estimate market_value_pre_accident based on make, model and year. "
                "Leave ground_truth fields to false since these will be ai generated jsons."
                "the goal isnt to get solely total losses , its to analyze as precisely as possible from the images the repair costs"
               
                "If the text file states engine:burn, Do not take it into consideration, it just means that the engine burns fuel"

            )


        print("Generating schema...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt] + uploaded_files,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            )
        )

        # Save the JSON output into the same folder
        output_path = os.path.join(folder_path, "aischema.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Success! Saved JSON schema to {output_path}")

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")

    finally:
        # Clean up files from Google's servers after processing to save your storage quota
        for uf in uploaded_files:
            try:
                client.files.delete(name=uf.name)
            except Exception:
                pass

def main(root_directory):
    vehicle_counter = 1

    # Traverse all folders and subfolders
    for root, dirs, files in os.walk(root_directory):
        # Check if the folder has any media files (skip the top-level parent if it's just folders)
        has_media = any(f.lower().endswith(('.jpg', '.jpeg', '.webp', '.png')) for f in files)

        if has_media:
            process_vehicle_folder(root, vehicle_counter)
            vehicle_counter += 1

            # Brief pause to respect API rate limits (15 RPM for the free tier)
            time.sleep(4)

if __name__ == "__main__":
    # If the script is placed inside the main folder, '.' refers to the current directory
    ROOT_DIR = "D:\Hunor\EGYETEM\A3.felev\Allamvizsga+tesztek\AutokReportok\Turners"
    main(ROOT_DIR)
