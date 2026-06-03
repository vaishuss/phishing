import pandas as pd

from sklearn.preprocessing import MinMaxScaler


INPUT_PATH = "data/processed/featured_dataset.csv"

OUTPUT_PATH = "data/processed/normalized_dataset.csv"


def normalize_features():

    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    print("\nOriginal Dataset:\n")

    print(df.head())

    # Separate features and labels
    X = df.drop("label", axis=1)

    y = df["label"]

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # Convert back to dataframe
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # Add labels back
    X_scaled_df["label"] = y

    # Save normalized dataset
    X_scaled_df.to_csv(OUTPUT_PATH, index=False)

    print("\nNormalization Completed")

    print(f"\nNormalized dataset saved to:\n{OUTPUT_PATH}")

    print("\nFirst 5 Rows:\n")

    print(X_scaled_df.head())


if __name__ == "__main__":
    normalize_features()