# AI Vehicle Damage Assessment

**Sapientia Hungarian University of Transylvania | Informatics Thesis**

A research project comparing state-of-the-art vision AI models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Flash) for automated vehicle damage assessment and repair cost estimation.

---

## Overview

| Property | Value |
|---|---|
| Duration | 12 weeks · 4 hours/day · ~336 hours |
| Models | GPT-4o Vision · Claude 3.5 Sonnet · Gemini 1.5 Flash |
| Experiments | 3 focused experiments |
| Est. Total Cost | ~$30 USD |

---

## Project Structure

```
vehicle-damage-ai/
├── data/
│   ├── raw/              # Scraped images — never modified
│   │   └── vehicle_NNN/
│   │       ├── img_01.jpg  img_02.jpg  ...
│   │       └── ground_truth.json
│   ├── test/             # 300 test vehicles for experiments
│   ├── metadata.csv      # One row per vehicle
│   └── repair_costs.csv  # Historical repair cost reference data
├── outputs/
│   ├── gpt4o/            # AI-generated reports per vehicle
│   ├── claude/
│   └── gemini/
├── scores/               # Judge scores + comparison CSVs
├── graphs/               # Auto-generated charts for thesis
├── cache/                # Cached API responses (cost savings)
│   ├── gpt4o/
│   ├── claude/
│   └── gemini/
├── run_experiment.py     # Main pipeline — EXP-1/2/3
├── judge.py              # LLM judge scoring (uses Gemini free tier)
├── analyze_results.py    # Generate tables + graphs
├── app.py                # Streamlit demo UI (defense)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
cd vehicle-damage-ai
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

### 3. Add your vehicle data

Place vehicle images in `data/test/vehicle_001/`, `data/test/vehicle_002/`, etc.
Optionally add `ground_truth.json` to `data/raw/vehicle_NNN/` for evaluation.

Update `data/metadata.csv` with one row per vehicle:
```
vehicle_id,image_count,has_ground_truth,damage_category,split
vehicle_001,5,true,front_collision,test
```

### 4. Run experiments

```bash
# EXP-1: All 3 models on full test set
python run_experiment.py

# EXP-2: Image count comparison (1, 3, 5, all images)
python run_experiment.py --max-images 1
python run_experiment.py --max-images 3
python run_experiment.py --max-images 5

# EXP-3: Prompt style comparison
python run_experiment.py --prompt simple
python run_experiment.py --prompt detailed

# Single vehicle demo (e.g. for defense)
python run_experiment.py --model claude --vehicle demo_car
```

### 5. Score outputs with LLM judge

```bash
python judge.py          # All models
python judge.py --model claude
```

### 6. Generate graphs

```bash
python analyze_results.py
```

Graphs are saved to `graphs/` as both `.png` (thesis) and `.html` (live demo).

### 7. Run the Streamlit demo (optional, for defense)

```bash
streamlit run app.py
```

---

## Experiments

| Experiment | What Is Tested | Expected Insight |
|---|---|---|
| **EXP-1** Model Comparison | GPT-4o vs. Claude vs. Gemini on 300 vehicles | One model wins on score + cost tradeoff |
| **EXP-2** Image Count | 1 vs. 3 vs. 5 vs. all images per vehicle | More angles → better accuracy |
| **EXP-3** Prompt Style | Simple vs. detailed structured prompt | Detailed prompt improves cost estimates |

---

## Metrics

| Metric | Description |
|---|---|
| Judge Score (avg 1-5) | Weighted LLM judge: part detection (35%) + cost (35%) + total loss (30%) |
| Cost MAE (EUR) | Mean absolute error of repair cost estimates |
| Total Loss Accuracy (%) | Correct repairable vs. total-loss decisions |
| Token Cost (USD) | API spend per 100 vehicles |
| Latency (seconds) | Average response time per vehicle |

---

## Graphs Generated

1. **Bar chart** — Average judge score per model (main EXP-1 result)
2. **Bar chart** — Cost estimation MAE per model
3. **Line chart** — Judge score vs. number of images (EXP-2)
4. **Bar chart** — API cost per 100 vehicles
5. **Scatter plot** — Predicted vs. actual repair cost

---

## 12-Week Roadmap

| Weeks | Focus |
|---|---|
| 1–2 | Data preparation — folder structure, metadata.csv, ground truth |
| 3–4 | Claude pipeline — 10 vehicles → full test set |
| 5 | Add GPT-4o and Gemini |
| 6 | LLM judge scoring on all outputs |
| 7 | EXP-1 analysis + graphs |
| 8 | EXP-2 + EXP-3 |
| 9 | All thesis figures + statistical tests |
| 10–11 | Write methodology + results chapters |
| 12 | Defense preparation |

---

## Research Questions

| Question | Answered by |
|---|---|
| How accurately can AI models detect vehicle damage? | EXP-1 judge scores + part detection F1 |
| How well can they estimate repair costs? | MAE in EUR vs. ground truth |
| Can AI determine total loss correctly? | Total loss accuracy % per model |
| Does more images improve accuracy? | EXP-2 line chart |
| Which model performs best overall? | Weighted ranking: score + cost + speed |
| How much does running the system cost? | Token cost table per 100 vehicles |

---

## The Golden Rule

> **Done is better than perfect.**
> A working system with 3 models and clean graphs beats a half-built complex agent every time.