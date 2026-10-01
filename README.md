# 🍄 Mushroom Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2)

An end-to-end machine learning pipeline that classifies mushrooms as **edible** or **poisonous** from their physical characteristics. The project covers data validation, preprocessing, baseline modelling, hyperparameter tuning, evaluation, model persistence, and experiment tracking with MLflow.

> **Disclaimer:** This is an educational project. Never use it to decide whether a real mushroom is safe to eat.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Experiment Tracking](#experiment-tracking)
- [Generated Files](#generated-files)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

The project follows a standard machine learning lifecycle:

1. **Load and validate** the raw dataset.
2. **Preprocess** features: remove constant columns, encode the target, one-hot encode categoricals.
3. **Split** the data into training and test sets.
4. **Train baseline classifiers** and compare them.
5. **Tune** the strongest models (Random Forest and Decision Tree) with GridSearchCV.
6. **Evaluate** using precision, recall, F1 score, and confusion matrices.
7. **Save** the final model and **log** parameters, metrics, and artifacts with MLflow.

The best-performing model in this repository is a **tuned Random Forest classifier**.

## Features

- Modular code: preprocessing, training, evaluation, validation, and model registry live in separate modules under `src/`
- Data validation before training
- Cross-validated comparison of several baseline models
- Hyperparameter tuning with GridSearchCV
- Saved classification reports, confusion matrices, and metrics as JSON
- MLflow experiment tracking and model registration
- Reproducible runs through two pipeline scripts, plus an exploratory notebook

## Tech Stack

| Area                | Tools                         |
|---------------------|-------------------------------|
| Language            | Python 3.10+                  |
| Data handling       | pandas, numpy                 |
| Modelling           | scikit-learn                  |
| Visualization       | matplotlib                    |
| Model persistence   | joblib                        |
| Experiment tracking | MLflow                        |
| Exploration         | Jupyter Notebook              |

## Dataset

The project uses the classic **Mushroom dataset** (`data/raw/mushroom.csv`). Every feature is categorical and describes the mushroom's appearance and habitat, such as cap shape, cap surface, gill color, odor, and habitat.

The target column is `class`:

| Value | Meaning   | Encoded as |
|-------|-----------|------------|
| `e`   | Edible    | `1`        |
| `p`   | Poisonous | `0`        |

Preprocessing removes constant features and one-hot encodes the remaining categorical variables. The processed train and test arrays are stored in `data/processed/`, along with `dataset_metadata.json`.

## Project Structure

```text
mushroom_classification/
├── data/
│   ├── raw/                     # mushroom.csv
│   └── processed/               # train/test arrays and dataset metadata
├── src/
│   ├── preprocess.py            # cleaning, encoding, train/test split
│   ├── train.py                 # model training and tuning
│   ├── evaluate.py              # metrics, reports, confusion matrices
│   ├── validate.py              # data validation checks
│   └── model_registry.py        # MLflow logging and model registration
├── pipelines/
│   ├── run_lab3_baseline.py     # baseline pipeline
│   └── run_lab4_tracking.py     # tuning and tracking pipeline
├── notebooks/
│   └── mushroom_classification.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git

### Installation

```bash
git clone https://github.com/Mokshagnna/mushroom_classification.git
cd mushroom_classification

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Run the commands from the project root with the virtual environment activated.

### 1. Baseline pipeline

```bash
python pipelines/run_lab3_baseline.py
```

Loads and validates the raw data, preprocesses it, trains baseline models, and saves the best one together with its evaluation outputs.

### 2. Tuning and tracking pipeline

```bash
python pipelines/run_lab4_tracking.py
```

Builds on the baseline run by adding:

- cross-validation comparison across baseline models
- Random Forest and Decision Tree tuning via GridSearchCV
- saved classification reports and confusion matrices
- MLflow run logging and model registration
- metadata export to `outputs/` and `artifacts/`

### 3. Notebook

For step-by-step exploration and visualization:

```bash
jupyter notebook notebooks/mushroom_classification.ipynb
```

## Models and Evaluation

Several baseline classifiers are compared first, then the strongest candidates, **Random Forest** and **Decision Tree**, are tuned with GridSearchCV.

**Why these metrics?** The positive class is *edible (1)*. A false positive means a poisonous mushroom is predicted as edible, which is the costly error in this problem. For that reason, models are judged on **precision** and **recall** as well as accuracy, and confusion matrices are saved for every evaluated model.

After running the pipelines, results are available in:

| File                                    | Contents                          |
|-----------------------------------------|-----------------------------------|
| `outputs/baseline_metrics.json`         | Baseline model metrics            |
| `outputs/rf_metrics.json`               | Tuned Random Forest metrics       |
| `outputs/rf_classification_report.txt`  | Per-class precision, recall, F1   |
| `outputs/rf_confusion_matrix.json`      | Confusion matrix                  |
| `outputs/tracking_results.json`         | Tracking run summary              |
| `outputs/model_registry_summary.json`   | Registered model details          |

## Experiment Tracking

MLflow records parameters, metrics, and artifacts for each run, giving a reproducible history of experiments. To browse runs in the MLflow UI, run this from the project root:

```bash
mlflow ui
```

Then open <http://127.0.0.1:5000> in your browser.

## Generated Files

The following folders are created when you run the pipelines. They are listed in `.gitignore`, so they will not be present after cloning the repository:

```text
models/       # baseline_best_model.joblib, tuned_random_forest.joblib
outputs/      # metrics, reports, confusion matrices, tracking summaries
artifacts/    # MLflow artifacts and run metadata
logs/         # pipeline logs
```

Run the pipelines (see [Usage](#usage)) to regenerate them.

## Future Improvements

- Add unit tests for the modules in `src/`
- Add a prediction script or a small Streamlit app for demos
- Rename the pipeline scripts to drop the `lab3` / `lab4` prefixes
- Add a CI workflow to run validation and tests on every push

## License

This project is intended for educational and research purposes.
