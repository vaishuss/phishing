import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATASET_PATH = "data/processed/normalized_dataset.csv"


def train_deep_learning_model():

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

    # Create MLP model
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    print("\nTraining Deep Learning Model...")

    # Train model
    model.fit(X_train, y_train)

    print("\nModel Training Completed")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    # Results
    print("\nDEEP LEARNING MODEL RESULTS\n")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(cm)


if __name__ == "__main__":
    train_deep_learning_model()