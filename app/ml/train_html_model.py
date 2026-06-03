import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATASET_PATH = "data/processed/html_features.csv"

MODEL_OUTPUT_PATH = (
    "data/processed/html_model.pkl"
)


def train_html_model():

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("\nHTML Dataset Loaded Successfully")

    # Features and labels
    X = df.drop(
        ["label", "file_id"],
        axis=1
    )

    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain-Test Split Completed")

    # Model
    model = LogisticRegression(
        max_iter=1000
    )

    print("\nTraining HTML Model...")

    # Train
    model.fit(X_train, y_train)

    print("\nTraining Completed")

    # SAVE MODEL
    joblib.dump(
        model,
        MODEL_OUTPUT_PATH
    )

    print(
        f"\nModel Saved To:\n{MODEL_OUTPUT_PATH}"
    )

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Results
    print("\nHTML MODEL RESULTS\n")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(cm)


if __name__ == "__main__":
    train_html_model()