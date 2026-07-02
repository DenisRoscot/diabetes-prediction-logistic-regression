# Diabetes Prediction with Logistic Regression

This project trains a logistic regression model to predict diabetes risk from tabular health data.

By default it uses scikit-learn's built-in diabetes dataset and converts the continuous disease progression score into a binary target:

- `0`: lower-risk progression
- `1`: higher-risk progression

You can also train it with a CSV dataset such as the Pima Indians Diabetes dataset. If the CSV has an `Outcome` column, that column is used as the target. Otherwise, the last column is used.

## Project Structure

```text
.
├── main.py            # CLI for training and prediction
├── pyproject.toml     # Python dependencies and script config
├── uv.lock            # uv lock file
└── README.md
```

Generated files are written to `artifacts/` and are ignored by Git.

## Setup

```bash
uv sync
```

## Train the Model

Train with the built-in demo dataset:

```bash
uv run python3 main.py
```

Train with your own CSV:

```bash
uv run python3 main.py --data data/diabetes.csv
```

The script saves:

- `artifacts/diabetes_logistic_regression.joblib`
- `artifacts/metrics.json`

## Make a Prediction

After training, pass feature values in the same order as the training columns:

```bash
uv run python3 main.py --predict 0.038 0.051 0.062 0.022 -0.044 -0.035 -0.043 -0.002 0.020 -0.018
```

## Model

The training pipeline uses:

- `StandardScaler`
- `LogisticRegression`
- `train_test_split` with stratification
- accuracy, confusion matrix, and classification report metrics

## Notes

The built-in dataset is useful for demonstrating the machine learning workflow locally. For a real medical prediction project, use a clinically appropriate dataset, validate data quality carefully, and do not treat this model as medical advice.
