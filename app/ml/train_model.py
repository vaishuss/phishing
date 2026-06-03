import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATASET_PATH = "data/processed/normalized_dataset.csv"

LOGISTIC_MODEL_PATH = "data/processed/logistic_model.pkl"

RF_MODEL_PATH = "data/processed/random_forest_model.pkl"


def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")

    print(f"{model_name} RESULTS")

    print(f"{'='*50}")

    print(f"\nAccuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(cm)


def train_models():

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset Loaded Successfully")

    # Features and labels
    X = df.drop("label", axis=1)

    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain-Test Split Completed")

    # ==================================================
    # Logistic Regression
    # ==================================================

    logistic_model = LogisticRegression(
        max_iter=1000
    )

    logistic_model.fit(X_train, y_train)

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH
    )

    evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "LOGISTIC REGRESSION"
    )

    # ==================================================
    # Random Forest
    # ==================================================

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    joblib.dump(
        rf_model,
        RF_MODEL_PATH
    )

    evaluate_model(
        rf_model,
        X_test,
        y_test,
        "RANDOM FOREST"
    )


if __name__ == "__main__":
    train_models()