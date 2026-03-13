"""
run_experiment.py — Main experiment runner for AI Vehicle Damage Assessment.

Usage:
    # Run all models on the full test set
    python run_experiment.py

    # Run a single model on the full test set
    python run_experiment.py --model claude
    python run_experiment.py --model gpt4o
    python run_experiment.py --model gemini

    # Run on a single vehicle (useful for demo / defense)
    python run_experiment.py --vehicle demo_car
    python run_experiment.py --model claude --vehicle demo_car

    # Control how many images per vehicle
    python run_experiment.py --max-images 1
    python run_experiment.py --max-images 3
    python run_experiment.py --max-images 5

    # Use an alternative prompt style (for EXP-3)
    python run_experiment.py --prompt detailed
"""

import os
import json
import base64
import argparse
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Prompts (EXP-3: simple vs. detailed)
# ---------------------------------------------------------------------------

DAMAGE_PROMPT_SIMPLE = """
Analyze this damaged vehicle. Return ONLY valid JSON:
{
  "damaged_parts": [{"part": "...", "severity": "light/moderate/heavy", "action": "repair/replace"}],
  "total_cost_eur": 0,
  "is_total_loss": false,
  "confidence": "low/medium/high"
}
"""

DAMAGE_PROMPT_DETAILED = """
You are an expert automotive damage assessor with 20 years of experience.
Analyze the provided vehicle photos carefully.
Return ONLY a valid JSON object — no explanation, no markdown:
{
  "vehicle_overview": "brief one-sentence description",
  "damaged_parts": [
    {
      "part": "front_bumper",
      "severity": "light | moderate | heavy | destroyed",
      "repair_action": "repair | repaint | replace",
      "confidence": "low | medium | high"
    }
  ],
  "visible_structural_damage": true,
  "airbags_deployed": false,
  "estimated_damage_zones": ["front", "left_side"],
  "total_cost_eur": 0,
  "is_total_loss": false,
  "confidence": "low/medium/high"
}
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
DATA = BASE / "data" / "test"
OUTPUTS = BASE / "outputs"
CACHE = BASE / "cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image(path: Path) -> str:
    """Return base64-encoded image content."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_images(vehicle_dir: Path, max_images: int) -> list[Path]:
    """Return up to *max_images* image paths from *vehicle_dir*."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        images.extend(vehicle_dir.glob(ext))
    return sorted(images)[:max_images]


def save_output(data: dict, output_dir: Path, vehicle_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{vehicle_id}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False)
    )


def save_cache(data: dict, cache_dir: Path, vehicle_id: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{vehicle_id}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False)
    )


def load_cache(cache_dir: Path, vehicle_id: str) -> dict | None:
    cache_file = cache_dir / f"{vehicle_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def parse_json_response(text: str) -> dict:
    """Extract the first JSON object from *text* (strip markdown fences)."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Cost context helper (used by all models)
# ---------------------------------------------------------------------------


def get_cost_context(damaged_parts: list) -> str:
    costs_csv = BASE / "data" / "repair_costs.csv"
    if not costs_csv.exists():
        return ""
    df = pd.read_csv(costs_csv)
    lines = []
    for item in damaged_parts:
        part = item.get("part", "")
        action = item.get("repair_action", item.get("action", "repair"))
        mask = (df["part"] == part) & (df["repair_action"] == action)
        match = df[mask]
        if not match.empty:
            row = match.iloc[0]
            lines.append(
                f"{part} ({action}): avg {row['avg_eur']} EUR, "
                f"range {row['min_eur']}-{row['max_eur']} EUR"
            )
        else:
            # Try any action for the part
            match_part = df[df["part"] == part]
            if not match_part.empty:
                row = match_part.iloc[0]
                lines.append(
                    f"{part} ({action}): avg {row['avg_eur']} EUR, "
                    f"range {row['min_eur']}-{row['max_eur']} EUR"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------


def call_claude(images: list[Path], prompt: str) -> dict:
    """Call Claude 3.5 Sonnet with vision."""
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    content = []
    for img_path in images:
        ext = img_path.suffix.lower().lstrip(".")
        media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encode_image(img_path),
                },
            }
        )
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return parse_json_response(response.content[0].text)


def call_gpt4o(images: list[Path], prompt: str) -> dict:
    """Call GPT-4o with vision."""
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    content = [{"type": "text", "text": prompt}]
    for img_path in images:
        ext = img_path.suffix.lower().lstrip(".")
        media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        b64 = encode_image(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            }
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return parse_json_response(response.choices[0].message.content)


def call_gemini(images: list[Path], prompt: str) -> dict:
    """Call Gemini 1.5 Flash with vision."""
    import google.generativeai as genai  # noqa: PLC0415

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    parts = [prompt]
    for img_path in images:
        from PIL import Image  # noqa: PLC0415

        parts.append(Image.open(img_path))

    response = model.generate_content(parts)
    return parse_json_response(response.text)


MODEL_CALLERS = {
    "claude": call_claude,
    "gpt4o": call_gpt4o,
    "gemini": call_gemini,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_vehicle(
    vehicle_dir: Path,
    model: str,
    max_images: int,
    prompt: str,
    dry_run: bool = False,
) -> dict | None:
    vehicle_id = vehicle_dir.name
    cache_dir = CACHE / model
    output_dir = OUTPUTS / model

    # Check cache first — always skip if already processed
    cached = load_cache(cache_dir, vehicle_id)
    if cached is not None:
        print(f"  [cache] {vehicle_id} — skipping (already done)")
        return cached

    images = load_images(vehicle_dir, max_images)
    if not images:
        print(f"  [skip]  {vehicle_id} — no images found")
        return None

    if dry_run:
        print(f"  [dry]   {vehicle_id} — would call {model} with {len(images)} image(s)")
        return None

    print(f"  [run]   {vehicle_id} — calling {model} with {len(images)} image(s) ...")
    t0 = time.time()
    try:
        caller = MODEL_CALLERS[model]
        result = caller(images, prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] {vehicle_id} — {exc}")
        return None

    elapsed = time.time() - t0
    result["_meta"] = {
        "vehicle_id": vehicle_id,
        "model": model,
        "image_count": len(images),
        "latency_s": round(elapsed, 2),
    }

    save_cache(result, cache_dir, vehicle_id)
    save_output(result, output_dir, vehicle_id)
    print(f"  [done]  {vehicle_id} — {elapsed:.1f}s")
    return result


def run_model(
    model: str,
    max_images: int,
    prompt: str,
    vehicle_filter: str | None,
    dry_run: bool,
) -> None:
    print(f"\n=== Model: {model}  max_images={max_images} ===")

    if vehicle_filter:
        vehicle_dirs = [DATA / vehicle_filter]
        if not vehicle_dirs[0].exists():
            print(f"[error] Vehicle directory not found: {vehicle_dirs[0]}")
            return
    else:
        vehicle_dirs = sorted(d for d in DATA.iterdir() if d.is_dir())

    if not vehicle_dirs:
        print("[warn] No vehicle directories found in data/test/")
        return

    for vehicle_dir in vehicle_dirs:
        run_vehicle(vehicle_dir, model, max_images, prompt, dry_run=dry_run)

    print(f"=== Done: {model} ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Vehicle Damage Assessment — Experiment Runner"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CALLERS.keys()),
        default=None,
        help="Which model to run (default: all three)",
    )
    parser.add_argument(
        "--vehicle",
        default=None,
        help="Run on a single vehicle directory name (e.g. demo_car)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Max images per vehicle (default: 5)",
    )
    parser.add_argument(
        "--prompt",
        choices=["simple", "detailed"],
        default="detailed",
        help="Prompt style: simple or detailed (default: detailed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without calling any API",
    )
    args = parser.parse_args()

    prompt_text = (
        DAMAGE_PROMPT_SIMPLE if args.prompt == "simple" else DAMAGE_PROMPT_DETAILED
    )

    models = [args.model] if args.model else list(MODEL_CALLERS.keys())
    for model in models:
        run_model(
            model=model,
            max_images=args.max_images,
            prompt=prompt_text,
            vehicle_filter=args.vehicle,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
