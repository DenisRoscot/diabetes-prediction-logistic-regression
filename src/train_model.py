import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
RANDOM_STATE = 42
COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def load_dataset():
    print("Loading dataset from URL.")
    return pd.read_csv(DATA_URL, names=COLUMN_NAMES)


def validate_dataset(data):
    missing_columns = set(COLUMN_NAMES) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Dataset is empty.")


def split_features_and_target(data):
    X = data.drop(columns="Outcome")
    y = data["Outcome"]
    return X, y


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def initialize_model():
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return model


def train_model(model, X_train_scaled, y_train):
    print("Training Logistic Regression model.")
    model.fit(X_train_scaled, y_train)
    return model


def make_predictions(model, X_test_scaled):
    print("Making predictions on the testing data.")
    predictions = model.predict(X_test_scaled)
    return predictions


def print_dataset_summary(data):
    print("\nDataset shape:")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nTarget distribution:")
    print(data["Outcome"].value_counts().sort_index())


def print_evaluation(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nAccuracy Score:")
    print(round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


def print_sample_inference(model, scaler, X_test):
    patient = X_test.iloc[[0]]
    patient_scaled = scaler.transform(patient)

    risk_probability = model.predict_proba(patient_scaled)[0][1]
    prediction = model.predict(patient_scaled)[0]
    label = "Positive Diagnosis" if prediction == 1 else "Control/Healthy"

    print("\nSample inference:")
    print(patient.to_string(index=False))
    print(f"Predicted class: {prediction} ({label})")
    print(f"Risk probability: {risk_probability:.4f}")


def main():
    data = load_dataset()
    validate_dataset(data)
    X, y = split_features_and_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = initialize_model()
    model = train_model(model, X_train_scaled, y_train)
    predictions = make_predictions(model, X_test_scaled)

    print_dataset_summary(data)
    print_evaluation(y_test, predictions)
    print_sample_inference(model, scaler, X_test)


if __name__ == "__main__":
    main()
