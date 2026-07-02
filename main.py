from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "diabetes_logistic_regression.joblib"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "metrics.json"


def load_training_data(csv_path: Path | None) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load a diabetes classification dataset.

    If no CSV is provided, scikit-learn's diabetes progression dataset is used as
    a local demo and converted into a binary high-risk target by median split.
    """
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data = pd.read_csv(csv_path)
        target_column = "Outcome" if "Outcome" in data.columns else data.columns[-1]
        features = data.drop(columns=[target_column])
        target = data[target_column].astype(int)
        source = f"CSV dataset: {csv_path} (target column: {target_column})"
        return features, target, source

    diabetes = load_diabetes(as_frame=True)
    features = diabetes.data
    target = (diabetes.target > diabetes.target.median()).astype(int)
    source = "scikit-learn diabetes demo dataset, target binarized above median progression"
    return features, target, source


def train_model(features: pd.DataFrame, target: pd.Series) -> tuple[Pipeline, dict[str, object]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["low_risk", "high_risk"],
            output_dict=True,
        ),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": list(features.columns),
    }
    return model, metrics


def save_artifacts(model: Pipeline, metrics: dict[str, object], model_path: Path, metrics_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def predict_sample(model_path: Path, values: list[float]) -> None:
    model: Pipeline = joblib.load(model_path)
    scaler = model.named_steps["scaler"]
    feature_names = list(scaler.feature_names_in_)
    expected_features = len(feature_names)

    if len(values) != expected_features:
        raise ValueError(f"Expected {expected_features} values, received {len(values)}")

    sample = pd.DataFrame([np.array(values, dtype=float)], columns=feature_names)
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    label = "high risk" if prediction == 1 else "low risk"

    print(f"Prediction: {label}")
    print(f"High-risk probability: {probability:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use a logistic regression model for diabetes risk prediction."
    )
    parser.add_argument("--data", type=Path, help="Optional CSV dataset. Uses 'Outcome' as target if present.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--predict",
        nargs="+",
        type=float,
        help="Feature values for a single prediction, in the same order as the training columns.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.predict:
        predict_sample(args.model_path, args.predict)
        return

    features, target, source = load_training_data(args.data)
    model, metrics = train_model(features, target)
    metrics["data_source"] = source
    save_artifacts(model, metrics, args.model_path, args.metrics_path)

    print(f"Data source: {source}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.metrics_path}")


if __name__ == "__main__":
    main()
