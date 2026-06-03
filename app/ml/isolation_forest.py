import pandas as pd

from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATASET_PATH = "data/processed/normalized_dataset.csv"


def run_isolation_forest():

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset Loaded Successfully")

    # Features and labels
    X = df.drop("label", axis=1)

    y = df["label"]

    # Create Isolation Forest model
    model = IsolationForest(
        contamination=0.48,
        random_state=42
    )

    print("\nTraining Isolation Forest Model...")

    # Train model
    model.fit(X)

    # Predict anomalies
    predictions = model.predict(X)

    # Convert predictions:
    # IsolationForest:
    # -1 = anomaly
    #  1 = normal

    # Our dataset:
    # 0 = phishing
    # 1 = legitimate

    mapped_predictions = []

    for pred in predictions:

        if pred == -1:
            mapped_predictions.append(0)
        else:
            mapped_predictions.append(1)

    # Evaluation
    accuracy = accuracy_score(
        y,
        mapped_predictions
    )

    precision = precision_score(
        y,
        mapped_predictions
    )

    recall = recall_score(
        y,
        mapped_predictions
    )

    f1 = f1_score(
        y,
        mapped_predictions
    )

    cm = confusion_matrix(
        y,
        mapped_predictions
    )

    # Results
    print("\nISOLATION FOREST RESULTS\n")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(cm)


if __name__ == "__main__":
    run_isolation_forest()