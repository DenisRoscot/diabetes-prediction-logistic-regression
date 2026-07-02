# Patient Diabetes Risk Prediction Engine

Production-grade binary classification engine designed to predict diabetes risk in patients based on clinical diagnostic measurements. This repository contains the data processing pipeline, feature normalization, and localized inference modeling built on top of Scikit-Learn's Logistic Regression architecture.

## Business & Technical Context
Early intervention reduces patient health degradation. This micro-service accepts tabular clinical diagnostic vectors and yields a high-velocity risk probability score (\(0.0 \dots 1.0\)) to support clinical decision-making.

### Clinical Feature Schema
The system models inference vectors using the following verified medical data parameters:
* `Pregnancies`: Total gestation cycles.
* `Glucose`: Plasma glucose concentration (2-hour oral glucose tolerance test profile).
* `BloodPressure`: Diastolic blood pressure metrics (mm Hg).
* `SkinThickness`: Triceps skin fold thickness matrix (mm).
* `Insulin`: 2-Hour serum insulin concentrations (mu U/ml).
* `BMI`: Body Mass Index profile (\(kg/m^2\)).
* `DiabetesPedigreeFunction`: Computed genetic diabetes history coefficient.
* `Age`: Structural age bracket.
* **Target (`Outcome`)**: Boolean indicator (`0` = Control/Healthy, `1` = Positive Diagnosis).

---

## Technical Pipeline Architecture

### Data Pipeline & Preprocessing
* **Ingestion Layer**: Robust decoupled tabular intake pipelines powered by `pandas`.
* **Data Splitting**: Stratified data splitting protocol (\(75\%\) Train / \(25\%\) Holdout Evaluation) ensuring structural validation balances.
* **Feature Scaling**: Robust feature normalization using a fitted `StandardScaler` pipeline to eliminate scaling bias across heterogeneous medical metrics.

### Modeling Stack
* **Algorithm**: Optimized `Logistic Regression` classifier.
* **Convergence Parameters**: Maximum iteration constraints scaled to `1000` steps to safely achieve mathematical convergence under dense data footprints.
* **Reproducibility**: Complete pseudo-random seed lockdown to isolate and stabilize model weights across training runs.

---

## Repository Structure
```text
diabetes-prediction-logistic-regression/
│
├── src/                   # Production Core Source Engine
│   └── train_model.py     # End-to-end model pipeline execution module
├── .gitignore             # Local development runtime exclusion filters
├── README.md              # System Architecture & Documentation
└── requirements.txt       # Production dependency configuration manifest
```

## System Deployment & Execution Guide

### 1. Environment Initialization
Isolate your localized dependencies using a Python virtual environment manager:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running Pipeline Inferences
To execute the end-to-end extraction, normalization, training, and evaluation runtime layer, trigger the production module:
```bash
python src/train_model.py
```

The dataset is loaded directly from a public CSV URL, so no local `data/` directory is required.

### 3. Core Validation Metrics (Baseline Model)
The pipeline prints the following validation metrics:
* **Accuracy Score**: Holdout-set classification accuracy.
* **Inference Logging**: Outputs a comprehensive classification tracking matrix (Precision, Recall, F1) alongside a raw Confusion Matrix for true/false positive isolation.
