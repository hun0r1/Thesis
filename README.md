# Thesis

This repository contains the data, scripts, and evaluation results used in my thesis on AI-based vehicle damage assessment.

## Repository Structure

- **Cars and Reports** — Contains vehicle damage reports, AI-generated damage analyses, and ground truth JSON files. Each vehicle also includes an outdated AI-based evaluation JSON file comparing the AI-generated results against the ground truth.

- **Graphs** — Contains visualizations of evaluation results, scores, vehicle data, and damage analysis statistics.

- **Metadata** — Contains a CSV file with one entry per vehicle, including its model, year, ground truth coverage, number of images used for AI analysis, and evaluation scores. This information is also stored in a SQLite database.

- **Results and Statistics** — Contains structured evaluation metrics and summary statistics.

- **Scores** — Contains per-vehicle evaluation scores in CSV format and JSON summaries organized by category, such as `score_parts` and `score_cost`. Evaluation is fully script-based: values from the AI-generated JSON files are compared against ground truth values and assigned scores from 1 to 5.

- **Scripts** — Contains the scripts used throughout the processing pipeline, from collecting vehicle data and generating AI damage reports to comparing reports against ground truth and calculating evaluation scores.