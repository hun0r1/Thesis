"""
STEP 1 — Setup Project Folder Structure (Gemini Only)
----------------------------------------------------
Run this ONCE to create the project layout.

Usage:
    python 01_setup_folders.py
"""

from pathlib import Path

# ── CHANGE THIS to where you want the project ──────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\Hunor\vehicle-damage-ai")
# ───────────────────────────────────────────────────────────────────────────

folders = [
    "data/test",            # test images
    "data/metadata",        # CSV metadata

    "outputs/gemini",       # Gemini AI outputs (JSON)

    "cache/gemini",         # Raw Gemini API responses

    "scores/gemini",        # Evaluation scores

    "graphs",               # Charts
    "notebooks",            # Jupyter notebooks
    "agent",                # Final agent script
]

def setup():
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
        print(f"  created: {folder}")

    # Create .env template (Gemini only)
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        env_file.write_text(
            "GOOGLE_API_KEY=your_key_here\n"
        )
        print("\n  created: .env  ← add your Gemini API key here")

    # Create .gitignore
    gitignore = PROJECT_ROOT / ".gitignore"
    gitignore.write_text(
        ".env\n"
        "cache/\n"
        "data/raw/\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".venv/\n"
    )
    print("  created: .gitignore")

    print(f"\nDone. Project created at: {PROJECT_ROOT}")
    print("\nNext step: run 02_build_metadata.py")

if __name__ == "__main__":
    setup()