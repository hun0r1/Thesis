"""
app.py — Streamlit UI for AI Vehicle Damage Assessment (defense demo).

Run with:
    streamlit run app.py

Features:
- Upload one or more vehicle photos
- Choose model (Claude / GPT-4o / Gemini)
- Analyze damage and display structured JSON report
- Show cost breakdown table
- Show judge score if ground truth is available
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Vehicle Damage Assessment",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Import pipeline helpers from run_experiment
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent


def _load_run_experiment():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_experiment", BASE / "run_experiment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🚗 AI Vehicle Damage Assessment")
st.markdown(
    "Upload vehicle photos and get a structured damage report using "
    "state-of-the-art vision models."
)

# Sidebar — model selection
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "AI Model",
        ["claude", "gpt4o", "gemini"],
        format_func=lambda m: {
            "claude": "Claude 3.5 Sonnet",
            "gpt4o": "GPT-4o Vision",
            "gemini": "Gemini 1.5 Flash (Free)",
        }[m],
    )
    prompt_style = st.radio("Prompt style", ["detailed", "simple"])
    max_images = st.slider("Max images to send", 1, 10, 5)

    st.markdown("---")
    st.markdown("**API Keys** (from environment or entered below)")

    if model == "claude":
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    elif model == "gpt4o":
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = st.text_input(
            "Google API Key",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

# Main area — image upload
uploaded_files = st.file_uploader(
    "Upload vehicle photos (JPG / PNG)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files[:4]):
        with cols[i % 4]:
            st.image(f, caption=f.name, use_container_width=True)

    if len(uploaded_files) > 4:
        st.info(f"… and {len(uploaded_files) - 4} more uploaded.")

analyze_btn = st.button("🔍 Analyze Damage", type="primary", disabled=not uploaded_files)

if analyze_btn and uploaded_files:
    with st.spinner(f"Calling {model} …"):
        try:
            re = _load_run_experiment()

            prompt_text = (
                re.DAMAGE_PROMPT_SIMPLE
                if prompt_style == "simple"
                else re.DAMAGE_PROMPT_DETAILED
            )

            # Save uploads to temp files
            tmp_paths = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files[:max_images]:
                    tmp_path = Path(tmpdir) / f.name
                    tmp_path.write_bytes(f.read())
                    tmp_paths.append(tmp_path)

                caller = re.MODEL_CALLERS[model]
                result = caller(tmp_paths, prompt_text)

        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Error: {exc}")
            st.stop()

    st.success("✅ Analysis complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Damage Report")
        parts = result.get("damaged_parts", [])
        if parts:
            df_parts = pd.DataFrame(parts)
            st.dataframe(df_parts, use_container_width=True)
        else:
            st.info("No damaged parts detected.")

        total_cost = result.get("total_cost_eur", "N/A")
        is_total_loss = result.get("is_total_loss", False)
        confidence = result.get("confidence", "N/A")

        col1a, col1b, col1c = st.columns(3)
        col1a.metric("Total Cost (EUR)", f"€{total_cost:,}" if isinstance(total_cost, (int, float)) else total_cost)
        col1b.metric("Total Loss", "🔴 YES" if is_total_loss else "🟢 NO")
        col1c.metric("Confidence", str(confidence).capitalize())

    with col2:
        st.subheader("📄 Raw JSON Output")
        st.json(result)

    # Cost breakdown from repair_costs.csv
    costs_csv = BASE / "data" / "repair_costs.csv"
    if costs_csv.exists() and parts:
        st.subheader("💶 Cost Reference (from historical data)")
        cost_context = re.get_cost_context(parts)
        if cost_context:
            st.text(cost_context)

    # Download button
    st.download_button(
        "⬇️ Download Report (JSON)",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="damage_report.json",
        mime="application/json",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AI Vehicle Damage Assessment · Sapientia Hungarian University of Transylvania "
    "· Informatics Thesis"
)
